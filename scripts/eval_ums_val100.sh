#!/bin/bash

# python tools/train_net.py.py --config-file configs/urchininf/
# sbatch --partition=accel scripts/saga_slurm_train_net.sh 


# Base plus synthetic aug
# Squidle
seed=1234575
python tools/train_net.py --eval-only --config-file configs/urchininf/Base-RCNN-FPN-urchininf_strongaug_ema_sq.yaml  SEED ${seed} OUTPUT_DIR outputs/urchininf/urchininf_baseline_strongaug_ema_sqval100/ LOGGING.GROUP_TAGS Inf2SQval100,UMS5VAL \
MODEL.WEIGHTS outputs/urchininf/urchininf_baseline_strongaug_ema_sqval100/squidle_urchin_2009_train_labelled_model_best_${seed}.pth
python tools/train_net.py --eval-only --config-file configs/urchininf/Base-RCNN-FPN-urchininf_strongaug_ema.yaml SEED ${seed} OUTPUT_DIR outputs/urchininf/urchininf_baseline_strongaug_ema_uddval100/ LOGGING.GROUP_TAGS Inf2UDDval100,UMS5VAL \
MODEL.WEIGHTS outputs/urchininf/urchininf_baseline_strongaug_ema_uddval100/UDD_train_labelled_model_best_${seed}.pth

# Best model strongaugEMA
# ALDI
python tools/train_net.py --eval-only --config-file configs/urchininf/ALDI-urchininf_sq.yaml SEED ${seed} OUTPUT_DIR outputs/urchininf/aldi_inf_sqval100_max/ LOGGING.GROUP_TAGS Inf2SQval100,UMS5VAL,BestP \
MODEL.WEIGHTS outputs/urchininf/aldi_inf_sqval100_max/squidle_urchin_2009_train_labelled_model_best_${seed}.pth
python tools/train_net.py --eval-only --config-file configs/urchininf/ALDI-urchininf.yaml SEED ${seed} OUTPUT_DIR outputs/urchininf/aldi_inf_uddval100_max/ LOGGING.GROUP_TAGS Inf2UDDval100,UMS5VAL,BestP \
MODEL.WEIGHTS outputs/urchininf/aldi_inf_uddval100_max/UDD_train_labelled_model_best_${seed}.pth

# MT _ To be done
python tools/train_net.py --eval-only --config-file configs/urchininf/MeanTeacher-urchininf_sq.yaml SEED ${seed} OUTPUT_DIR outputs/urchininf/mt_inf_sqval100_max/ LOGGING.GROUP_TAGS Inf2SQval100,UMS5VAL,BestP \
MODEL.WEIGHTS outputs/urchininf/mt_inf_sqval100_max/squidle_urchin_2009_train_labelled_model_best_${seed}.pth
python tools/train_net.py --eval-only --config-file configs/urchininf/MeanTeacher-urchininf.yaml SEED ${seed} OUTPUT_DIR outputs/urchininf/mt_inf_uddval100_max/ LOGGING.GROUP_TAGS Inf2UDDval100,UMS5VAL,BestP \
MODEL.WEIGHTS outputs/urchininf/mt_inf_uddval100_max/UDD_train_labelled_model_best_${seed}.pth

#python tools/train_net.py --eval-only --config-file configs/urchininf/OracleT-RCNN-FPN-urchininf_strongaug_ema.yaml OUTPUT_DIR outputs/urchininf/oracle_strongaug_ema_sq/ LOGGING.GROUP_TAGS Sq2UDD,UMS7 \
#MODEL.WEIGHTS outputs/urchininf/oracle_strongaug_ema_sq/model_final.pth