#!/bin/bash

# python tools/train_net.py --config-file configs/urchininf/
# sbatch --partition=accel --exclude=c7-7 tools/saga_slurm_train_net.sh 
run="SAOD5"
lr=0.02
SEEDS=(1234575 2234575 3234575)
sbatch --partition=accel --exclude=c7-7 tools/saga_slurm_train_net.sh ../coco/CoStudent-RCNN-FPN-coco.yaml SEED 1234575 OUTPUT_DIR outputs/TEST/coco_easy LOGGING.GROUP_TAGS TEST,COCO,EASY MODEL.WEIGHTS outputs/TEST/coco_easy_0.01/model_0019999.pth SOLVER.STEPS (21000, 24000,)