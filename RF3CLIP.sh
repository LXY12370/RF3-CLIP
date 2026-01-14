#!/bin/bash
DATASET_DIR=/home/cds/Robust_Fairness_for_Medical_Image-main/data
RESULT_DIR=.
MODEL_ARCH=vit-b16 # Options: vit-b16 | vit-l14
NUM_EPOCH=10
MODALITY_TYPE='slo_fundus'
ATTRIBUTE_TYPE=race # Options: race | gender | ethnicity | language
SUMMARIZED_NOTE_FILE=gpt4_summarized_notes.csv
LR=1e-5
BATCH_SIZE=32
LAMBDA=1e-7
BATCH_SIZE_FAIR=32

PERF_FILE=${MODEL_ARCH}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_FairCLIP.csv

python ./RF3CLIP.py \
		--num_epochs ${NUM_EPOCH} \
		--dataset_dir ${DATASET_DIR} \
		--result_dir ${RESULT_DIR}/results/glaucoma_RobustFairCLIP_${MODEL_ARCH}_${ATTRIBUTE_TYPE}_ \
		--lr ${LR} \
		--batch_size ${BATCH_SIZE} \
		--perf_file ${PERF_FILE} \
		--model_arch ${MODEL_ARCH} \
		--attribute ${ATTRIBUTE_TYPE} \
		--summarized_note_file ${SUMMARIZED_NOTE_FILE} 
