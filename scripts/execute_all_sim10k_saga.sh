#!/bin/bash

# python tools/train_net.py --config-file configs/urchininf/
# sbatch --partition=accel scripts/saga_slurm_train_net.sh 


jobid1=$(sbatch --partition=accel scripts/saga_slurm_train_net.sh ../sim10k/Base-RCNN-FPN-Sim10k_strongaug_ema.yaml "SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS1" | awk '{print $4}')
echo "Submitted job1.sh with Job ID: $jobid1"
jobid2=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../sim10k/ALDI-Sim10k_bestpre.yaml "SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS1" | awk '{print $4}')
echo "Submitted job1.sh with Job ID: $jobid2"
jobid3=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../sim10k/MeanTeacher-Sim10k_bestpre.yaml "SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS1" | awk '{print $4}')
echo "Submitted job1.sh with Job ID: $jobid3"

jobid4=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../sim10k/ALDI-Sim10k.yaml "SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS1" | awk '{print $4}')
echo "Submitted job1.sh with Job ID: $jobid2"
jobid5=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../sim10k/MeanTeacher-Sim10k.yaml "SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS1" | awk '{print $4}')
echo "Submitted job1.sh with Job ID: $jobid3"