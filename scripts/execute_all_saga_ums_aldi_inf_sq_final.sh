#!/bin/bash

# python tools/train_net.py --config-file configs/urchininf/
# sbatch --partition=accel scripts/saga_slurm_train_net.sh 


# Base plus synthetic aug
# Squidle
seed=3234575
jobid1=$(sbatch --partition=accel scripts/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema_sq.yaml  "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/urchininf_baseline_strongaug_ema_sq/\' LOGGING.GROUP_TAGS Inf2SQ,UMS5 " | awk '{print $4}')
echo "Submitted job1.sh with Job ID: $jobid1 "

# Final model

# ALDI 
sbatch --partition=accel scripts/saga_slurm_train_net.sh ALDI-urchininf_sq.yaml SEED 3234575 OUTPUT_DIR outputs/urchininf/aldi_inf_sq_final LOGGING.GROUP_TAGS Inf2SQ,UMS5 MODEL.WEIGHTS outputs/urchininf/urchininf_baseline_strongaug_ema_sq/model_final.pth
