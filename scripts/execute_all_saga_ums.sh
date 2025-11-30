#!/bin/bash

# python tools/train_net.py --config-file configs/urchininf/
# sbatch --partition=accel scripts/saga_slurm_train_net.sh 


# Base plus synthetic aug
# Squidle
seed=3234575
jobid1=$(sbatch --partition=accel scripts/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema_sq.yaml  "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/urchininf_baseline_strongaug_ema_sq/\' LOGGING.GROUP_TAGS Inf2SQ,UMS5 " | awk '{print $4}')
jobid2=$(sbatch --partition=accel scripts/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema.yaml "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/urchininf_baseline_strongaug_ema_udd/\' LOGGING.GROUP_TAGS Inf2UDD,UMS5 " | awk '{print $4}')
echo "Submitted job1.sh with Job ID: $jobid1 and $jobid2"

# Final model
# MT
sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh MeanTeacher-urchininf_sq.yaml "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/mt_inf_sq_final/\' LOGGING.GROUP_TAGS Inf2SQ,UMS5 MODEL.WEIGHTS \'outputs/urchininf/urchininf_baseline_strongaug_ema_sq/model_final.pth\'"
sbatch --dependency=afterok:${jobid2} --partition=accel scripts/saga_slurm_train_net.sh MeanTeacher-urchininf.yaml "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/mt_inf_udd_final/\' LOGGING.GROUP_TAGS Inf2UDD,UMS5 MODEL.WEIGHTS \'outputs/urchininf/urchininf_baseline_strongaug_ema_udd/model_final.pth\'"

# ALDI 
sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ALDI-urchininf_sq.yaml "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/aldi_inf_sq_final/\' LOGGING.GROUP_TAGS Inf2SQ,UMS5 MODEL.WEIGHTS \'outputs/urchininf/urchininf_baseline_strongaug_ema_sq/model_final.pth\'"
sbatch --dependency=afterok:${jobid2} --partition=accel scripts/saga_slurm_train_net.sh ALDI-urchininf.yaml "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/aldi_inf_udd_final/\' LOGGING.GROUP_TAGS Inf2UDD,UMS5 MODEL.WEIGHTS \'outputs/urchininf/urchininf_baseline_strongaug_ema_udd/model_final.pth\'"

# Best model strongaugEMA
# ALDI
sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ALDI-urchininf_sq.yaml "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/aldi_inf_sq_max/\' LOGGING.GROUP_TAGS Inf2SQ,UMS5,BestP MODEL.WEIGHTS \'outputs/urchininf/urchininf_baseline_strongaug_ema_sq/squidle_urchin_2011_test_model_best.pth\'"
sbatch --dependency=afterok:${jobid2} --partition=accel scripts/saga_slurm_train_net.sh ALDI-urchininf.yaml "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/aldi_inf_udd_max/\' LOGGING.GROUP_TAGS Inf2UDD,UMS5,BestP MODEL.WEIGHTS \'outputs/urchininf/urchininf_baseline_strongaug_ema_udd/UDD_test_model_best.pth\'"

# MT _ To be done
sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh MeanTeacher-urchininf_sq.yaml "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/mt_inf_sq_max/\' LOGGING.GROUP_TAGS Inf2SQ,UMS5,BestP MODEL.WEIGHTS \'outputs/urchininf/urchininf_baseline_strongaug_ema_sq/squidle_urchin_2011_test_model_best.pth\'"
sbatch --dependency=afterok:${jobid2} --partition=accel scripts/saga_slurm_train_net.sh MeanTeacher-urchininf.yaml "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/mt_inf_udd_max/\' LOGGING.GROUP_TAGS Inf2UDD,UMS5,BestP MODEL.WEIGHTS \'outputs/urchininf/urchininf_baseline_strongaug_ema_udd/UDD_test_model_best.pth\'"

# UMS Best model strongaugEMA
# ALDI
sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ALDI-urchininf_sq_ums.yaml "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/aldi_inf_sq_ums/\' LOGGING.GROUP_TAGS Inf2SQ,UMS5,UMS MODEL.WEIGHTS \'outputs/urchininf/urchininf_baseline_strongaug_ema_sq/squidle_urchin_train_umsdas_ioukl_model_best.pth\'"
sbatch --dependency=afterok:${jobid2} --partition=accel scripts/saga_slurm_train_net.sh ALDI-urchininf_udd_ums.yaml "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/aldi_inf_udd_ums/\' LOGGING.GROUP_TAGS Inf2UDD,UMS5,UMS MODEL.WEIGHTS \'outputs/urchininf/urchininf_baseline_strongaug_ema_udd/UDD_train_umsdas_ioukl_model_best.pth\'"

# MT _ 
sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh MeanTeacher-urchininf_sq_ums.yaml "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/mt_inf_sq_ums/\' LOGGING.GROUP_TAGS Inf2SQ,UMS5,UMS MODEL.WEIGHTS \'outputs/urchininf/urchininf_baseline_strongaug_ema_sq/squidle_urchin_train_umsdas_ioukl_model_best.pth\'"
sbatch --dependency=afterok:${jobid2} --partition=accel scripts/saga_slurm_train_net.sh MeanTeacher-urchininf_udd_ums.yaml "SEED ${seed} OUTPUT_DIR \'outputs/urchininf/mt_inf_udd_ums/\' LOGGING.GROUP_TAGS Inf2UDD,UMS5,UMS MODEL.WEIGHTS \'outputs/urchininf/urchininf_baseline_strongaug_ema_udd/UDD_train_umsdas_ioukl_model_best.pth\'"


# Results if adding 100 labelled target to training.

sbatch --partition=accel tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema.yaml "SEED 1234575 OUTPUT_DIR \'outputs/urchininf/base_strongaug_ema_inf_udd100/\' LOGGING.GROUP_TAGS Inf+100tgt2UDD,T5  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split_100\',\) "
sbatch --partition=accel tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema_sq.yaml "SEED 1234575 OUTPUT_DIR \'outputs/urchininf/base_strongaug_ema_inf_sq100/\' LOGGING.GROUP_TAGS Inf+100tgt2SQ,T5  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split_100\',\) "


sbatch --partition=accel tools/saga_slurm_train_net.sh MeanTeacher-urchininf_sq.yaml "SEED 1234575 OUTPUT_DIR \'outputs/urchininf/mt_inf_sq100/\' LOGGING.GROUP_TAGS Inf+100tgt2SQ,T5 DATASETS.UNLABELED \(\'squidle_east_tas_urchins_train\',\) DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split_100\',\) "
sbatch --partition=accel tools/saga_slurm_train_net.sh MeanTeacher-urchininf.yaml "SEED 1234575 OUTPUT_DIR \'outputs/urchininf/mt_inf_udd100/\' LOGGING.GROUP_TAGS Inf+100tgt2UDD,T5 DATASETS.UNLABELED \(\) DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split_100\',\) "


sbatch --partition=accel tools/saga_slurm_train_net.sh ALDI-urchininf_sq.yaml "SEED 1234575 OUTPUT_DIR \'outputs/urchininf/aldi_inf_sq100/\' LOGGING.GROUP_TAGS Inf+100tgt2SQ,T5 DATASETS.UNLABELED \(\'squidle_east_tas_urchins_train\',\) DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split_100\',\) "
sbatch --partition=accel tools/saga_slurm_train_net.sh ALDI-urchininf.yaml "SEED 1234575 OUTPUT_DIR \'outputs/urchininf/aldi_inf_udd100/\' LOGGING.GROUP_TAGS Inf+100tgt2UDD,T5 DATASETS.UNLABELED \(\) DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split_100\',\) "


# Results if using 100 labelled for evaluation
