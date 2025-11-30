#!/bin/bash

# python tools/train_net.py --config-file configs/urchininf/
# sbatch --partition=accel scripts/saga_slurm_train_net.sh 

seed=1234575
#jobid1=$(sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/Base-RCNN-FPN-imosauv_strongaug_ema.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
#echo "Submitted job1.sh with Job ID: $jobid1"
#jobid2=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_ums.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
#jobid3=$(sbatch  --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_ums.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')

#jobid4=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_final.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
#jobid5=$(sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_final.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')

#jobid6=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_max.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
#jobid7=$(sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_max.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
#echo "Submitted seed ${seed} job ids MT UMS $jobid3 "

seed=2234575
#jobid1=$(sbatch --dependency=afterok:${jobid3} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/Base-RCNN-FPN-imosauv_strongaug_ema.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
echo "Submitted job1.sh with Job ID: $jobid1"
#jobid2=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_ums.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
#jobid3=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_ums.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')

#jobid4=$(sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_final.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
#jobid5=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_final.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')

#jobid6=$(sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_max.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
#jobid7=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_max.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
#echo "Submitted seed ${seed} job ids ALDI UMS $jobid2 MT UMS $jobid3 ALDI Final $jobid4 MT Final $jobid5 ALDI Msc $jobid6 MT Max $jobid7"
#echo "Submitted seed ${seed} job ids  ALDI Final $jobid4 ALDI Msc $jobid6 "

seed=3234575
jobid1=$(sbatch --dependency=afterok:16121938:16121939 --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/Base-RCNN-FPN-imosauv_strongaug_ema.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
echo "Submitted job1.sh with Job ID: $jobid1"
jobid2=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_ums.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
jobid3=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_ums.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')

jobid4=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_final.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
jobid5=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_final.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')

jobid6=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_max.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
jobid7=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_max.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
echo "Submitted seed ${seed} job ids ALDI UMS $jobid2 MT UMS $jobid3 ALDI Final $jobid4 MT Final $jobid5 ALDI Msc $jobid6 MT Max $jobid7"

Submitted job1.sh with Job ID: 16121950
Submitted seed 3234575 job ids ALDI UMS 16121951 MT UMS 16121952 ALDI Final 16121953 MT Final 16121954 ALDI Msc 16121955 MT Max 16121956
