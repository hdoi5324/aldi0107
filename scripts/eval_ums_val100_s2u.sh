#!/bin/bash

# python tools/train_net.py.py --config-file configs/imosauv/
# sbatch --partition=accel scripts/saga_slurm_train_net.sh 


# Base plus synthetic aug
# Squidle
for seed in 1234575 2234575 3234575
do

  #python tools/train_net.py --eval-only --config-file configs/imosauv/Base-RCNN-FPN-imosauv_strongaug_ema.yaml SEED ${seed} OUTPUT_DIR outputs/imosauv/imosauv_baseline_strongaug_ema_uddval100_${seed}/ LOGGING.GROUP_TAGS Inf2UDDval100,UMS7VAL \
  #MODEL.WEIGHTS outputs/imosauv/imosauv_baseline_strongaug_ema_uddval100_${seed}/UDD_train_labelled_model_best.pth
  
  # Best model strongaugEMA
  # ALDI
  
  python tools/train_net.py --eval-only --config-file configs/imosauv/ALDI-imosauv_final.yaml SEED ${seed} OUTPUT_DIR outputs/imosauv/aldi_sq_uddval100_max_${seed}/ LOGGING.GROUP_TAGS Inf2UDDval100,UMS7VAL,BestP \
  MODEL.WEIGHTS outputs/imosauv/aldi_sq_uddval100_max_${seed}/UDD_train_labelled_model_best.pth
  
  # MT _ To be done
  
  python tools/train_net.py --eval-only --config-file configs/imosauv/MeanTeacher-imosauv_final.yaml SEED ${seed} OUTPUT_DIR outputs/imosauv/mt_sq_uddval100_max_${seed}/ LOGGING.GROUP_TAGS Inf2UDDval100,UMS7VAL,BestP \
  MODEL.WEIGHTS outputs/imosauv/mt_sq_uddval100_max_${seed}/UDD_train_labelled_model_best.pth
done
