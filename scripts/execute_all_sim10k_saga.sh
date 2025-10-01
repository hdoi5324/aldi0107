#!/bin/bash

# python tools/train_net.py --config-file configs/urchininf/
# sbatch --partition=accel scripts/saga_slurm_train_net.sh 


jobid1=$(sbatch --partition=accel scripts/saga_slurm_train_net.sh ../sim10k/Base-RCNN-FPN-Sim10k_strongaug_ema.yaml "SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS2"
echo "Submitted job1.sh with Job ID: $jobid1"
sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../sim10k/ALDI-Sim10k_ums.yaml "SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS2"
sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../sim10k/MeanTeacher-ums.yaml "SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS2"

sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../sim10k/ALDI-Sim10k.yaml "SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS2"
sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../sim10k/MeanTeacher-Sim10k.yaml "SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS2"

sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../sim10k/ALDI-Sim10k_max.yaml "SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS2"
sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../sim10k/MeanTeacher_max.yaml "SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS2"
