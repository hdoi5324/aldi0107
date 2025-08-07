#!/bin/bash

# python tools/train_net.py --config-file configs/urchininf/
# sbatch --partition=accel tools/saga_slurm_train_net.sh 

# Base - strongaug
#Squidle
sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/OracleWL-RCNN-FPN-redcup_strongaug_ema.yaml "SEED 1234575 LOGGING.GROUP_TAGS WL1 "
sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/OracleWL-RCNN-FPN-urchin_strongaug_ema.yaml "SEED 1234575 LOGGING.GROUP_TAGS WL1 "
