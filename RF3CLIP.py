import os
import sys
import io
import json
import time
import math
import random
import argparse
import numpy as np
import pandas as pd
from collections import Counter

from geomloss import SamplesLoss
from torch import Tensor

import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import wandb

sys.path.append('.')
from src.modules3 import *
from src import logger


# -----------------------------
# Utils
# -----------------------------
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def convert_models_to_fp32(model):
    """Safely convert params/grads to fp32 for optimizer.step() when using fp16 weights."""
    for p in model.parameters():
        if p is None:
            continue
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


# -----------------------------
# Scheme-A: Student-t IRLS per-sample weight
# -----------------------------
def t_weight_irls(z: torch.Tensor, nu: float, w_min: float = 0.0) -> torch.Tensor:
    """
    z: standardized residual tensor, shape [B]
    nu: degrees of freedom (>0), smaller => heavier tail => stronger down-weighting for outliers
    w_min: clamp minimum weight to avoid vanishing gradients
    """
    nu = float(nu)
    w = (nu + 1.0) / (nu + z * z + 1e-8)
    w = torch.clamp(w, min=float(w_min))
    return w


# -----------------------------
# RCS (classification-friendly): fuse TOKEN FEATURES (not attention maps)
# -----------------------------
class CLIPWithRCS(nn.Module):
    """
    Wrap OpenAI CLIP model; override encode_image with token-feature fusion.
    Optimized defaults: mid_layers=(9,10,11), lambda=0.2 for better semantic preservation.
    """
    def __init__(
        self,
        clip_model: nn.Module,
        mid_layers=(9, 10, 11), # Optimized from original (6,7,8,9,10)
        lambda_rcs: float = 0.2, # Optimized from original 0.5
        pool: str = "patch_mean",
    ):
        super().__init__()
        self.clip_model = clip_model
        self.mid_layers = set(list(mid_layers))
        self.lambda_rcs = float(lambda_rcs)
        assert pool in ["cls", "patch_mean"]
        self.pool = pool

        # expose attributes used elsewhere
        self.visual = self.clip_model.visual
        self.transformer = self.clip_model.transformer
        self.token_embedding = self.clip_model.token_embedding
        self.positional_embedding = self.clip_model.positional_embedding
        self.ln_final = self.clip_model.ln_final
        self.text_projection = self.clip_model.text_projection
        self.logit_scale = self.clip_model.logit_scale

    def encode_text(self, text):
        return self.clip_model.encode_text(text)

    def encode_image(self, image):
        visual = self.visual

        # ✅ ensure input dtype matches conv1 weight dtype (fp16 or fp32)
        target_dtype = visual.conv1.weight.dtype
        if image.dtype != target_dtype:
            image = image.to(dtype=target_dtype)

        x = visual.conv1(image)  # [B, width, grid, grid]
        b, c, gh, gw = x.shape
        x = x.reshape(b, c, gh * gw).permute(0, 2, 1)  # [B, N, width]

        # prepend CLS
        cls = visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(b, 1, x.shape[-1], device=x.device, dtype=x.dtype)
        x = torch.cat([cls, x], dim=1)  # [B, 1+N, width]

        # add positional embedding + pre LN
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        # transformer expects [T, B, C]
        x = x.permute(1, 0, 2)  # [T, B, C]

        mids = []
        for idx, blk in enumerate(visual.transformer.resblocks):
            x = blk(x)
            if idx in self.mid_layers:
                mids.append(x)

        x_last = x
        if len(mids) > 0 and self.lambda_rcs > 0:
            x_mid = torch.stack(mids, dim=0).mean(dim=0)  # [T, B, C]
            lam = self.lambda_rcs
            x = (1.0 - lam) * x_last + lam * x_mid
        else:
            x = x_last

        x = x.permute(1, 0, 2)  # [B, T, C]
        x = visual.ln_post(x)

        # pooling
        if self.pool == "cls":
            x_pool = x[:, 0, :]  # [B, C]
        else:
            x_pool = x[:, 1:, :].mean(dim=1)  # [B, C]

        if visual.proj is not None:
            x_pool = x_pool @ visual.proj  # [B, 512]
        return x_pool

    def forward(self, images, texts):
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


# -----------------------------
# Class-conditional Center (star-shaped) OT
# -----------------------------
class ClassConditionalCenter(nn.Module):
    def __init__(self, center_size: int = 64, ema_beta: float = 0.9, device: str = "cuda"):
        super().__init__()
        assert center_size > 0
        self.center_size = int(center_size)
        self.ema_beta = float(ema_beta)

        # buffers for y=0 and y=1
        self.register_buffer("center0", torch.zeros(self.center_size, 1, device=device))
        self.register_buffer("center1", torch.zeros(self.center_size, 1, device=device))
        self._initialized = False

    @torch.no_grad()
    def _sample_to_size(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x[:, None]
        n = x.shape[0]
        if n <= 0:
            return None
        if n >= self.center_size:
            idx = torch.randperm(n, device=x.device)[: self.center_size]
            return x[idx]
        # n < center_size: sample with replacement
        idx = torch.randint(low=0, high=n, size=(self.center_size,), device=x.device)
        return x[idx]

    @torch.no_grad()
    def maybe_init_from_batch(self, corr_y0: torch.Tensor, corr_y1: torch.Tensor):
        if self._initialized:
            return
        if corr_y0 is not None and corr_y0.numel() > 0:
            s0 = self._sample_to_size(corr_y0)
            if s0 is not None:
                self.center0.copy_(s0)
        if corr_y1 is not None and corr_y1.numel() > 0:
            s1 = self._sample_to_size(corr_y1)
            if s1 is not None:
                self.center1.copy_(s1)
        self._initialized = True

    @torch.no_grad()
    def ema_update(self, y: int, corr_y: torch.Tensor):
        if corr_y is None or corr_y.numel() == 0:
            return
        s = self._sample_to_size(corr_y)
        if s is None:
            return
        beta = self.ema_beta
        if int(y) == 0:
            self.center0.mul_(beta).add_(s * (1.0 - beta))
        else:
            self.center1.mul_(beta).add_(s * (1.0 - beta))

    def get_center(self, y: int) -> torch.Tensor:
        return self.center0 if int(y) == 0 else self.center1


def _safe_corr_points(image_features: torch.Tensor, text_features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    corr = (image_features * text_features).sum(dim=1).float()  # [B]
    corr = corr - corr.min()
    corr = corr + eps
    return corr[:, None]  # [B,1]


def _select_by_label(points: torch.Tensor, y: torch.Tensor, target: int) -> torch.Tensor:
    m = (y == int(target))
    if m.sum().item() == 0:
        return None
    return points[m]


# -----------------------------
# Eval
# -----------------------------
def evaluate_model(model, dataloader, device):
    """Evaluate model on given dataloader and return predictions, labels, attributes and loss"""
    eval_avg_loss = 0.0
    all_probs = []
    all_labels = []
    all_attrs = []

    for batch in dataloader:
        images, texts, label_and_attributes = batch

        images = images.to(device)
        texts = texts.to(device)
        glaucoma_labels = label_and_attributes[:, 0].to(device)
        attributes = label_and_attributes[:, 1:].to(device)

        class_text_feats = []
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            for i in range(texts.shape[1]):
                text_features = model.encode_text(texts[:, i, :])
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                class_text_feats.append(text_features[:, None, :])
            class_text_feats = torch.cat(class_text_feats, dim=1)

        vl_prob, vl_logits = compute_vl_prob(image_features, class_text_feats)

        all_probs.append(vl_prob[:, 1].detach().cpu().numpy())
        all_labels.append(glaucoma_labels.detach().cpu().numpy())
        all_attrs.append(attributes.detach().cpu().numpy())

        loss = F.binary_cross_entropy(vl_prob[:, 1].float(), glaucoma_labels.float())
        eval_avg_loss += float(loss.item())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_attrs = np.concatenate(all_attrs, axis=0)
    eval_avg_loss /= max(len(dataloader), 1)

    return all_probs, all_labels, all_attrs, eval_avg_loss


# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser(description='FairCLIP Training/Fine-Tuning')

parser.add_argument('--seed', default=-1, type=int, help='seed for initializing training.')
parser.add_argument('--num_epochs', default=20, type=int) # Increased default
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, metavar='LR', dest='lr') # Optimized Default
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=6e-5, type=float, metavar='W', dest='weight_decay')

parser.add_argument('--result_dir', default='./results', type=str)
parser.add_argument('--dataset_dir', default='./data', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--eval_set', default='test', type=str, help='options: val | test')
parser.add_argument('--summarized_note_file', default='', type=str)
parser.add_argument('--text_source', default='note', type=str, help='options: note | label')
parser.add_argument('--perf_file', default='', type=str)
parser.add_argument('--model_arch', default='vit-b16', type=str, help='options: vit-b16 | vit-l14')
parser.add_argument('--pretrained_weights', default='', type=str)
parser.add_argument('--attribute', default='race', type=str, help='race|gender|ethnicity|language')

# Batchsize fairloss: Optimized to 32 to prevent OOM during backprop
parser.add_argument('--batchsize_fairloss', default=32, type=int)
parser.add_argument('--lambda_fairloss', default=1e-4, type=float)
parser.add_argument('--sinkhorn_blur', default=1e-4, type=float)

# -----------------------------
# class-conditional + center star-shaped OT params
# -----------------------------
parser.add_argument('--center_size', default=64, type=int, help='Center buffer size per class (y=0/1)')
parser.add_argument('--center_beta', default=0.9, type=float, help='EMA beta for updating center')
parser.add_argument('--fair_warmup_epochs', default=3, type=int, help='Warmup epochs before enabling fairness loss')

# -----------------------------
# RCS params
# -----------------------------
parser.add_argument('--disable_rcs', action='store_true', help='Disable RCS (default: enabled)')
parser.add_argument('--lambda_rcs', default=0.2, type=float, help='RCS fusion weight (0~1)')
parser.add_argument('--rcs_layers', default='9,10,11', type=str,
                    help='Comma-separated mid layer indices. Default Optimized: 9,10,11')
parser.add_argument('--rcs_pool', default='patch_mean', type=str, choices=['cls', 'patch_mean'],
                    help='Pooling after token fusion')

# -----------------------------
# Scheme-A TWeight params
# -----------------------------
parser.add_argument('--disable_tweight', action='store_true', help='Disable Student-t IRLS reweighting (default: enabled)')
parser.add_argument('--t_nu', default=100.0, type=float, help='Student-t dof nu')
parser.add_argument('--t_w_min', default=0.85, type=float, help='Clamp minimum per-sample weight')
parser.add_argument('--t_eps', default=1e-6, type=float, help='Epsilon for std stabilization')
parser.add_argument('--t_warmup_epochs', default=2, type=int, help='Warmup epochs before enabling tweight')

# -----------------------------
# NEW: Group DRO
# -----------------------------
parser.add_argument('--disable_dro', action='store_true', help='Disable Group DRO')


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    args = parser.parse_args()

    args.seed = 822
    set_random_seed(args.seed)
    logger.log(f'===> random seed: {args.seed}')

    logger.configure(dir=args.result_dir, log_suffix='train')

    os.makedirs(args.result_dir, exist_ok=True)
    with open(os.path.join(args.result_dir, f'args_train.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # attribute info
    groups_in_attrs = [3, 2, 2, 3]
    attr_to_idx = {'race': 0, 'gender': 1, 'ethnicity': 2, 'language': 3}
    model_arch_mapping = {'vit-b16': 'ViT-B/16', 'vit-l14': 'ViT-L/14'}

    best_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'best_{args.perf_file}')

    # --- RESTORED: Full Metric Header Generation ---
    if args.perf_file != '':
        if not os.path.exists(best_global_perf_file):
            auc_head_str = ''
            dpd_head_str = ''
            eod_head_str = ''
            esacc_head_str = ''
            esauc_head_str = ''
            group_disparity_head_str = ''
            for i in range(len(groups_in_attrs)):
                auc_head_str += ', '.join([f'auc_attr{i}_group{x}' for x in range(groups_in_attrs[i])]) + ', '
            dpd_head_str += ', '.join([f'dpd_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            eod_head_str += ', '.join([f'eod_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            esacc_head_str += ', '.join([f'esacc_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            esauc_head_str += ', '.join([f'esauc_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            group_disparity_head_str += ', '.join(
                [f'std_group_disparity_attr{x}, max_group_disparity_attr{x}' for x in range(len(groups_in_attrs))]
            ) + ', '

            with open(best_global_perf_file, 'w') as f:
                f.write(
                    f'epoch, acc, {esacc_head_str} auc, {esauc_head_str} '
                    f'{auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_head_str} path\n'
                )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base_model, preprocess = clip.load(model_arch_mapping[args.model_arch], device=device, jit=False)

    # ---- RCS Configuration ----
    use_rcs = (not args.disable_rcs)
    if use_rcs:
        mid_layers = tuple(int(x.strip()) for x in args.rcs_layers.split(",") if x.strip() != "")
        model = CLIPWithRCS(
            base_model,
            mid_layers=mid_layers,
            lambda_rcs=args.lambda_rcs,
            pool=args.rcs_pool,
        ).to(device)
        logger.log(f"[RCS] ENABLED. mid_layers={sorted(list(mid_layers))}, lambda={args.lambda_rcs}, pool={args.rcs_pool}")
    else:
        model = base_model
        logger.log("[RCS] Disabled.")

    # ---- Config Logging ----
    use_tweight = (not args.disable_tweight)
    if use_tweight:
        logger.log(f"[TWeight] ENABLED. warmup={args.t_warmup_epochs}, nu={args.t_nu}, w_min={args.t_w_min}")
    
    use_dro = (not args.disable_dro)
    if use_dro:
        logger.log(f"[Group DRO] ENABLED. Attribute: {args.attribute}")

    # datasets
    train_dataset = fair_vl_med_dataset(
        args.dataset_dir, preprocess, subset='Training',
        text_source=args.text_source, summarized_note_file=args.summarized_note_file
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )

    val_dataset = fair_vl_med_dataset(args.dataset_dir, preprocess, subset='Validation')
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )

    test_dataset = fair_vl_med_dataset(args.dataset_dir, preprocess, subset='Test')
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )

    logger.log(f'# of training samples: {len(train_dataset)}, # of validation samples: {len(val_dataset)}, # of testing samples: {len(test_dataset)}')

    # group dataloaders for fairness (by selected attribute)
    group_dataloaders = []
    for i in range(groups_in_attrs[attr_to_idx[args.attribute]]):
        tmp_dataset = fair_vl_group_dataset(
            args.dataset_dir, preprocess,
            text_source='note', summarized_note_file=args.summarized_note_file,
            attribute=args.attribute, thegroup=i
        )
        tmp_dataloader = DataLoader(
            tmp_dataset, batch_size=args.batchsize_fairloss, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=False
        )
        group_dataloaders.append(endless_loader(tmp_dataloader))

    group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(train_dataset)
    logger.log(f'group size on race in training set: {group_size_on_race}')
    logger.log(f'group size on gender in training set: {group_size_on_gender}')
    logger.log(f'group size on ethnicity in training set: {group_size_on_ethnicity}')

    # fp16 weights on GPU (OpenAI CLIP default)
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)

    # Sinkhorn OT
    loss_sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=args.sinkhorn_blur)

    # Class-conditional centers
    center_module = ClassConditionalCenter(
        center_size=args.center_size,
        ema_beta=args.center_beta,
        device=device
    ).to(device)
    logger.log(f"[FairCenter-CC] ENABLED. center_size={args.center_size}, beta={args.center_beta}")

    optimizer = optim.Adam(
        [
            {"params": model.transformer.parameters(), "lr": args.lr},
            {"params": model.visual.parameters(), "lr": args.lr},
        ],
        lr=args.lr,
        betas=(0.9, 0.94),
        eps=1e-6,
        weight_decay=args.weight_decay
    )

    best_epoch = 0
    best_auc = -1e18
    best_acc = -1e18
    best_es_acc = -1e18
    best_es_auc = -1e18
    best_auc_groups = None
    best_dpd_groups = None
    best_eod_groups = None
    best_between_group_disparity = None
    best_model_state = None

    num_batches = len(train_dataloader)
    memory_module = np.zeros((num_batches, args.num_epochs), dtype=np.float32)

    for epoch in range(args.num_epochs):
        model.train()
        avg_loss = 0.0

        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            images, texts, label_and_attributes = batch
            images = images.to(device)
            texts = texts.to(device)
            
            # Attributes for DRO
            all_attributes = label_and_attributes[:, 1:].to(device)
            current_attr_idx = attr_to_idx[args.attribute]
            
            glaucoma_labels = label_and_attributes[:, 0].to(device).long()

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            logits_per_image = (logit_scale * image_features @ text_features.t()).float()
            logits_per_text = logits_per_image.t().float()

            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            # ---------------------------------------
            # 1. Base Contrastive Loss (Per Sample)
            # ---------------------------------------
            ce_i = F.cross_entropy(logits_per_image, ground_truth, reduction='none')
            ce_t = F.cross_entropy(logits_per_text,  ground_truth, reduction='none')
            per_sample_loss = 0.5 * (ce_i + ce_t)  # [B]

            # ---------------------------------------
            # 2. Student-t Weighting (Optional: Noise Removal)
            # ---------------------------------------
            if use_tweight and (epoch >= int(args.t_warmup_epochs)):
                mu = per_sample_loss.mean().detach()
                sd = per_sample_loss.std(unbiased=False).detach() + float(args.t_eps)
                z = (per_sample_loss - mu) / sd
                w = t_weight_irls(z, nu=float(args.t_nu), w_min=float(args.t_w_min)).detach()
                w = w / (w.mean() + 1e-8)
                weighted_per_sample_loss = w * per_sample_loss
            else:
                weighted_per_sample_loss = per_sample_loss

            # ---------------------------------------
            # 3. Soft Group DRO (Fairness Enhancement)
            # ---------------------------------------
            contrastive_loss = None
            if use_dro and (epoch >= 3): # Add small warmup for DRO
                current_group_labels = all_attributes[:, current_attr_idx].long()
                unique_groups = torch.unique(current_group_labels)
                
                group_losses = []
                for g in unique_groups:
                    mask_g = (current_group_labels == g)
                    if mask_g.sum() > 0:
                        g_loss = weighted_per_sample_loss[mask_g].mean()
                        group_losses.append(g_loss)
                
                if len(group_losses) > 0:
                    stack_losses = torch.stack(group_losses)
                    dro_weights = F.softmax(stack_losses, dim=0)
                    contrastive_loss = torch.sum(dro_weights * stack_losses)
                else:
                    contrastive_loss = weighted_per_sample_loss.mean()
            else:
                contrastive_loss = weighted_per_sample_loss.mean()

            if i < num_batches:
                memory_module[i, epoch] = float(contrastive_loss.item())

            total_loss = contrastive_loss

            # ---------------------------------------
            # 4. Center (Star-shaped) OT Fairness Loss
            # ---------------------------------------
            corr_b_points = _safe_corr_points(image_features, text_features)
            
            with torch.no_grad():
                center_module.maybe_init_from_batch(_select_by_label(corr_b_points, glaucoma_labels, 0), _select_by_label(corr_b_points, glaucoma_labels, 1))
                center_module.ema_update(0, _select_by_label(corr_b_points, glaucoma_labels, 0))
                center_module.ema_update(1, _select_by_label(corr_b_points, glaucoma_labels, 1))

            if epoch >= int(args.fair_warmup_epochs) and args.lambda_fairloss > 0:
                fair_loss_val = 0.0
                valid_terms = 0

                for x in group_dataloaders:
                    images_dist, texts_dist, label_and_attributes_dist = next(x)
                    images_dist = images_dist.to(device)
                    texts_dist = texts_dist.to(device)
                    y_dist = label_and_attributes_dist[:, 0].to(device).long()

                    # <--- CRITICAL FIX: Removed 'with torch.no_grad():' --->
                    img_f = model.encode_image(images_dist)
                    txt_f = model.encode_text(texts_dist)
                    img_f = img_f / img_f.norm(dim=1, keepdim=True)
                    txt_f = txt_f / txt_f.norm(dim=1, keepdim=True)
                    corr_g_points = _safe_corr_points(img_f, txt_f)

                    c0 = center_module.get_center(0).detach()
                    corr_g_y0 = _select_by_label(corr_g_points, y_dist, 0)
                    if corr_g_y0 is not None and corr_g_y0.numel() > 0:
                        fair_loss_val = fair_loss_val + loss_sinkhorn(corr_g_y0, c0)
                        valid_terms += 1

                    c1 = center_module.get_center(1).detach()
                    corr_g_y1 = _select_by_label(corr_g_points, y_dist, 1)
                    if corr_g_y1 is not None and corr_g_y1.numel() > 0:
                        fair_loss_val = fair_loss_val + loss_sinkhorn(corr_g_y1, c1)
                        valid_terms += 1

                if valid_terms > 0:
                    fair_loss_val = fair_loss_val / float(valid_terms)
                    total_loss = total_loss + float(args.lambda_fairloss) * fair_loss_val

            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            avg_loss += float(total_loss.item())

        avg_loss /= max(len(train_dataloader), 1)

        # Evaluate on validation
        logger.log(f'Evaluating on validation set at epoch {epoch}...')
        val_probs, val_labels, val_attrs, val_avg_loss = evaluate_model(model, val_dataloader, device)

        val_overall_acc, val_eval_es_acc, val_overall_auc, val_eval_es_auc, val_eval_aucs_by_attrs, val_eval_dpds, val_eval_eods, val_between_group_disparity = \
            evalute_comprehensive_perf(val_probs, val_labels, val_attrs.T)

        logger.log(f'===> epoch[{epoch:03d}/{args.num_epochs:03d}], training loss: {avg_loss:.4f}, val loss: {val_avg_loss:.4f}, val AUC: {val_overall_auc:.4f}')

        if val_overall_auc > best_auc:
            best_auc = val_overall_auc
            best_acc = val_overall_acc
            best_epoch = epoch
            best_es_acc = val_eval_es_acc
            best_es_auc = val_eval_es_auc
            best_auc_groups = val_eval_aucs_by_attrs
            best_dpd_groups = val_eval_dpds
            best_eod_groups = val_eval_eods
            best_between_group_disparity = val_between_group_disparity
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_avg_loss,
                    'val_auc': val_overall_auc,
                },
                os.path.join(args.result_dir, "best_clip_model.pth")
            )
            logger.log(f'**** New best model saved at epoch {epoch} with val AUC: {val_overall_auc:.4f}')

        if args.result_dir is not None:
            np.savez(
                os.path.join(args.result_dir, f'val_pred_gt_ep{epoch:03d}.npz'),
                val_pred=val_probs, val_gt=val_labels, val_attr=val_attrs
            )

        # --- RESTORED: Full Validation Logging ---
        logger.logkv('epoch', epoch)
        logger.logkv('trn_loss', round(avg_loss, 4))
        logger.logkv('val_loss', round(val_avg_loss, 4))
        logger.logkv('val_acc', round(val_overall_acc, 4))
        logger.logkv('val_auc', round(val_overall_auc, 4))

        for ii in range(len(val_eval_es_acc)):
            logger.logkv(f'val_es_acc_attr{ii}', round(val_eval_es_acc[ii], 4))
        for ii in range(len(val_eval_es_auc)):
            logger.logkv(f'val_es_auc_attr{ii}', round(val_eval_es_auc[ii], 4))
        for ii in range(len(val_eval_aucs_by_attrs)):
            for iii in range(len(val_eval_aucs_by_attrs[ii])):
                logger.logkv(f'val_auc_attr{ii}_group{iii}', round(val_eval_aucs_by_attrs[ii][iii], 4))

        for ii in range(len(val_between_group_disparity)):
            logger.logkv(f'val_auc_attr{ii}_std_group_disparity', round(val_between_group_disparity[ii][0], 4))
            logger.logkv(f'val_auc_attr{ii}_max_group_disparity', round(val_between_group_disparity[ii][1], 4))

        for ii in range(len(val_eval_dpds)):
            logger.logkv(f'val_dpd_attr{ii}', round(val_eval_dpds[ii], 4))
        for ii in range(len(val_eval_eods)):
            logger.logkv(f'val_eod_attr{ii}', round(val_eval_eods[ii], 4))

        logger.dumpkvs()

    # -----------------------------
    # Final evaluation on test set
    # -----------------------------
    logger.log(f'=== Final Evaluation on Test Set ===')
    logger.log(f'Loading best model from epoch {best_epoch} with val AUC: {best_auc:.4f}')

    if best_model_state is not None:
        model.load_state_dict(best_model_state, strict=True)

    test_probs, test_labels, test_attrs, test_avg_loss = evaluate_model(model, test_dataloader, device)

    test_overall_acc, test_eval_es_acc, test_overall_auc, test_eval_es_auc, test_eval_aucs_by_attrs, test_eval_dpds, test_eval_eods, test_between_group_disparity = \
        evalute_comprehensive_perf(test_probs, test_labels, test_attrs.T)

    logger.log(f'Test Results - Loss: {test_avg_loss:.4f}, Acc: {test_overall_acc:.4f}, AUC: {test_overall_auc:.4f}')
    logger.log(f'Test AUC by groups: {test_eval_aucs_by_attrs}')

    if args.result_dir is not None:
        np.savez(
            os.path.join(args.result_dir, 'test_pred_gt_best.npz'),
            test_pred=test_probs, test_gt=test_labels, test_attr=test_attrs
        )

    # --- RESTORED: Full Test Logging ---
    logger.logkv('test_epoch', best_epoch)
    logger.logkv('test_loss', round(test_avg_loss, 4))
    logger.logkv('test_acc', round(test_overall_acc, 4))
    logger.logkv('test_auc', round(test_overall_auc, 4))

    for ii in range(len(test_eval_es_acc)):
        logger.logkv(f'test_es_acc_attr{ii}', round(test_eval_es_acc[ii], 4))
    for ii in range(len(test_eval_es_auc)):
        logger.logkv(f'test_es_auc_attr{ii}', round(test_eval_es_auc[ii], 4))
    for ii in range(len(test_eval_aucs_by_attrs)):
        for iii in range(len(test_eval_aucs_by_attrs[ii])):
            logger.logkv(f'test_auc_attr{ii}_group{iii}', round(test_eval_aucs_by_attrs[ii][iii], 4))

    for ii in range(len(test_between_group_disparity)):
        logger.logkv(f'test_auc_attr{ii}_std_group_disparity', round(test_between_group_disparity[ii][0], 4))
        logger.logkv(f'test_auc_attr{ii}_max_group_disparity', round(test_between_group_disparity[ii][1], 4))

    for ii in range(len(test_eval_dpds)):
        logger.logkv(f'test_dpd_attr{ii}', round(test_eval_dpds[ii], 4))
    for ii in range(len(test_eval_eods)):
        logger.logkv(f'test_eod_attr{ii}', round(test_eval_eods[ii], 4))

    logger.dumpkvs()

    # --- RESTORED: Appending to Perf File ---
    if args.perf_file != '':
        best_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'best_{args.perf_file}')
        if os.path.exists(best_global_perf_file):
            with open(best_global_perf_file, 'a') as f:
                esacc_head_str = ', '.join([f'{x:.4f}' for x in test_eval_es_acc]) + ', '
                esauc_head_str = ', '.join([f'{x:.4f}' for x in test_eval_es_auc]) + ', '

                auc_head_str = ''
                for i in range(len(test_eval_aucs_by_attrs)):
                    auc_head_str += ', '.join([f'{x:.4f}' for x in test_eval_aucs_by_attrs[i]]) + ', '

                group_disparity_str = ''
                for i in range(len(test_between_group_disparity)):
                    group_disparity_str += ', '.join([f'{x:.4f}' for x in test_between_group_disparity[i]]) + ', '

                dpd_head_str = ', '.join([f'{x:.4f}' for x in test_eval_dpds]) + ', '
                eod_head_str = ', '.join([f'{x:.4f}' for x in test_eval_eods]) + ', '

                path_str = f'{args.result_dir}_seed{args.seed}_testauc{test_overall_auc:.4f}'
                f.write(
                    f'{best_epoch}, {test_overall_acc:.4f}, {esacc_head_str} '
                    f'{test_overall_auc:.4f}, {esauc_head_str} '
                    f'{auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_str} {path_str}\n'
                )

    os.rename(args.result_dir, f'{args.result_dir}_seed{args.seed}_testauc{test_overall_auc:.4f}')
