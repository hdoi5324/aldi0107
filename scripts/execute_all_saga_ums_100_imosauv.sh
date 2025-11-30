#!/bin/bash

# python tools/train_net.py --config-file configs/imosauv/
# sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/


# Base plus synthetic aug
# Squidle
for seed in 1234575 2234575 3234575
do
  jobid2=$(sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/Base-RCNN-FPN-imosauv_strongaug_ema.yaml "SEED ${seed} OUTPUT_DIR \'outputs/imosauv/imosauv_baseline_strongaug_ema_udd100_${seed}/\' LOGGING.GROUP_TAGS SQ+100tgt2UDD,UMS7 \
  DATASETS.TRAIN \(\'squidle_urchin_train\',\'UDD_train_split_100\',\) " | awk '{print $4}')
  echo "Submitted job1.sh with Job ID: $jobid2"
  
  # Final model
  # MT
  sbatch --dependency=afterok:${jobid2} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_final.yaml "SEED ${seed} OUTPUT_DIR \'outputs/imosauv/mt_sq100tgt_udd_final_${seed}/\' LOGGING.GROUP_TAGS SQ+100tgt2UDD,UMS7 MODEL.WEIGHTS \'outputs/imosauv/imosauv_baseline_strongaug_ema_udd100_${seed}/model_final.pth\' \
  DATASETS.TRAIN \(\'squidle_urchin_train\',\'UDD_train_split_100\',\) "
  
  # ALDI 
  sbatch --dependency=afterok:${jobid2} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_final.yaml "SEED ${seed} OUTPUT_DIR \'outputs/imosauv/aldi_sq100tgt_udd_final_${seed}/\' LOGGING.GROUP_TAGS SQ+100tgt2UDD,UMS7 MODEL.WEIGHTS \'outputs/imosauv/imosauv_baseline_strongaug_ema_udd100_${seed}/model_final.pth\' \
  DATASETS.TRAIN \(\'squidle_urchin_train\',\'UDD_train_split_100\',\) "
  
  # UMS Best model strongaugEMA
  # ALDI
  sbatch --dependency=afterok:${jobid2} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_ums.yaml "SEED ${seed} OUTPUT_DIR \'outputs/imosauv/aldi_sq100tgt_udd_ums_${seed}/\' LOGGING.GROUP_TAGS SQ+100tgt2UDD,UMS7,UMS MODEL.WEIGHTS \'outputs/imosauv/imosauv_baseline_strongaug_ema_udd100_${seed}/UDD_train_umsdas_ioukl_model_best.pth\' \
  DATASETS.TRAIN \(\'squidle_urchin_train\',\'UDD_train_split_100\',\) "
  
  # MT _ 
  sbatch --dependency=afterok:${jobid2} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_ums.yaml "SEED ${seed} OUTPUT_DIR \'outputs/imosauv/mt_sq100tgt_udd_ums_${seed}/\' LOGGING.GROUP_TAGS SQ+100tgt2UDD,UMS7,UMS MODEL.WEIGHTS \'outputs/imosauv/imosauv_baseline_strongaug_ema_udd100_${seed}/UDD_train_umsdas_ioukl_model_best.pth\' \
  DATASETS.TRAIN \(\'squidle_urchin_train\',\'UDD_train_split_100\',\) "
done

Submitted job1.sh with Job ID: 16102863
Submitted batch job 16102864
Submitted batch job 16102865
Submitted batch job 16102866
Submitted batch job 16102867
Submitted job1.sh with Job ID: 16102868
Submitted batch job 16102869
Submitted batch job 16102870
Submitted batch job 16102871
Submitted batch job 16102872
Submitted job1.sh with Job ID: 16102873
Submitted batch job 16102874
Submitted batch job 16102875
Submitted batch job 16102876
Submitted batch job 16102877

