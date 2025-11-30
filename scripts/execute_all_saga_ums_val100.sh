#!/bin/bash

# python tools/train_net.py --config-file configs/urchininf/
# sbatch --partition=accel scripts/saga_slurm_train_net.sh 


# Base plus synthetic aug
# Squidle
seed=1234575
jobid1=$(sbatch --partition=accel scripts/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema_sq.yaml  "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/urchininf_baseline_strongaug_ema_sqval100/\' UMS.UNLABELED None LOGGING.GROUP_TAGS Inf2SQval100,UMS5 \
DATASETS.TEST \(\'squidle_urchin_2009_train_split_100\',\) " | awk '{print $4}')
jobid2=$(sbatch --partition=accel scripts/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema.yaml "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/urchininf_baseline_strongaug_ema_uddval100/\' UMS.UNLABELED None LOGGING.GROUP_TAGS Inf2UDDval100,UMS5 \
DATASETS.TEST \(\'UDD_train_split_100\',\) " | awk '{print $4}')
echo "Submitted job1.sh with Job ID: $jobid1 and $jobid2"

# Best model strongaugEMA
# ALDI
sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ALDI-urchininf_sq.yaml "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/aldi_inf_sqval100_max/\' UMS.UNLABELED None LOGGING.GROUP_TAGS Inf2SQval100,UMS5,BestP MODEL.WEIGHTS \'outputs/urchininf/urchininf_baseline_strongaug_ema_sqval100/squidle_urchin_2009_train_labelled_model_best.pth\' \
DATASETS.TEST \(\'squidle_urchin_2009_train_split_100\',\) "
sbatch --dependency=afterok:${jobid2} --partition=accel scripts/saga_slurm_train_net.sh ALDI-urchininf.yaml "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/aldi_inf_uddval100_max/\' UMS.UNLABELED None LOGGING.GROUP_TAGS Inf2UDDval100,UMS5,BestP MODEL.WEIGHTS \'outputs/urchininf/urchininf_baseline_strongaug_ema_uddval100/UDD_train_labelled_model_best.pth\' \
DATASETS.TEST \(\'UDD_train_split_100\',\) "

# MT _ To be done
sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh MeanTeacher-urchininf_sq.yaml "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/mt_inf_sqval100_max/\' UMS.UNLABELED None LOGGING.GROUP_TAGS Inf2SQval100,UMS5,BestP MODEL.WEIGHTS \'outputs/urchininf/urchininf_baseline_strongaug_ema_sqval100/squidle_urchin_2009_train_labelled_model_best.pth\' \
DATASETS.TEST \(\'squidle_urchin_2009_train_split_100\',\) "
sbatch --dependency=afterok:${jobid2} --partition=accel scripts/saga_slurm_train_net.sh MeanTeacher-urchininf.yaml "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/mt_inf_uddval100_max/\' UMS.UNLABELED None LOGGING.GROUP_TAGS Inf2UDDval100,UMS5,BestP MODEL.WEIGHTS \'outputs/urchininf/urchininf_baseline_strongaug_ema_uddval100/UDD_train_labelled_model_best.pth\' \
DATASETS.TEST \(\'UDD_train_split_100\',\) "

