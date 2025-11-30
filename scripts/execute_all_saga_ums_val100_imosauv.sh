#!/bin/bash

# python tools/train_net.py --config-file configs/imosauv/
# sbatch --partition=accel scripts/saga_slurm_train_net.sh 


# Base plus synthetic aug
# Squidle
for seed in 1234575 2234575 3234575
do
  jobid2=$(sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/Base-RCNN-FPN-imosauv_strongaug_ema.yaml "SEED ${seed} OUTPUT_DIR \'outputs/imosauv/imosauv_baseline_strongaug_ema_uddval100_${seed}/\' UMS.UNLABELED None LOGGING.GROUP_TAGS SQ2UDDval100,UMS7 \
  DATASETS.TEST \(\'UDD_train_split_100\',\) " | awk '{print $4}')
  echo "Submitted job1.sh with Job ID: $jobid1 and $jobid2"
  
  # Best model strongaugEMA
  # ALDI
  sbatch --dependency=afterok:${jobid2} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_max.yaml "SEED ${seed} OUTPUT_DIR \'outputs/imosauv/aldi_sq_uddval100_max_${seed}/\' UMS.UNLABELED None LOGGING.GROUP_TAGS SQ2UDDval100,UMS7,BestP MODEL.WEIGHTS \'outputs/imosauv/imosauv_baseline_strongaug_ema_uddval100_${seed}/UDD_train_labelled_model_best.pth\' \
  DATASETS.TEST \(\'UDD_train_split_100\',\) "
  
  # MT _ To be done
  sbatch --dependency=afterok:${jobid2} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_max.yaml "SEED ${seed} OUTPUT_DIR \'outputs/imosauv/mt_sq_uddval100_max_${seed}/\' UMS.UNLABELED None LOGGING.GROUP_TAGS SQ2UDDval100,UMS7,BestP MODEL.WEIGHTS \'outputs/imosauv/imosauv_baseline_strongaug_ema_uddval100_${seed}/UDD_train_labelled_model_best.pth\' \
  DATASETS.TEST \(\'UDD_train_split_100\',\) "
done