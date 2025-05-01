#!/bin/bash

# python tools/train_net.py --config-file configs/urchininf/
# sbatch --partition=accel tools/saga_slurm_train_net.sh 


sbatch --partition=accel tools/saga_slurm_train_net.sh ../sim10k/Base-RCNN-FPN-Sim10k.yaml "SEED 1234575 LOGGING.GROUP_TAGS S2C,T6 "
sbatch --partition=accel tools/saga_slurm_train_net.sh ../sim10k/Base-RCNN-FPN-Sim10k_strongaug_ema.yaml "SEED 1234575 LOGGING.GROUP_TAGS S2C,T6 "
sbatch --partition=accel tools/saga_slurm_train_net.sh ../sim10k/ALDI-Sim10k.yaml "SEED 1234575 LOGGING.GROUP_TAGS S2C,T6 "
sbatch --partition=accel tools/saga_slurm_train_net.sh ../sim10k/MeanTeacher-Sim10k.yaml "SEED 1234575 LOGGING.GROUP_TAGS S2C,T6 "
sbatch --partition=accel tools/saga_slurm_train_net.sh ../sim10k/ALDI-Sim10k_bestpre.yaml "SEED 1234575 LOGGING.GROUP_TAGS S2C,T6 "
sbatch --partition=accel tools/saga_slurm_train_net.sh ../sim10k/MeanTeacher-Sim10k_bestpre.yaml "SEED 1234575 LOGGING.GROUP_TAGS S2C,T6 "
