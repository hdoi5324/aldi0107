#!/bin/bash

# python tools/train_net.py --config-file configs/urchininf/
# sbatch --partition=accel scripts/saga_slurm_train_net.sh 

#seed=1234575
#jobid1=$(sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/Base-RCNN-FPN-imosauv_strongaug_ema.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
#echo "Submitted job1.sh with Job ID: $jobid1"
#jobid2=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_ums.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
jobid3=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_ums.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')

#jobid4=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_final.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
#jobid5=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_final.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')

#jobid6=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_max.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
#jobid7=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_max.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
#echo "Submitted job ids $jobid2 $jobid3 $jobid4 $jobid5 $jobid6 $jobid7"

seed=2234575
jobid1=$(sbatch --dependency=afterok:16094334:16094335:16094336:16094337:16094338:16094339 --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/Base-RCNN-FPN-imosauv_strongaug_ema.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
echo "Submitted job1.sh with Job ID: $jobid1"
jobid2=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_ums.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
jobid3=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_ums.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')

jobid4=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_final.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
jobid5=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_final.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')

jobid6=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_max.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
jobid7=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_max.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
echo "Submitted job ids $jobid2 $jobid3 $jobid4 $jobid5 $jobid6 $jobid7"

seed=3234575
jobid1=$(sbatch --dependency=afterok:${jobid2}:${jobid3}:${jobid4}:${jobid5}:${jobid6}:${jobid7} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/Base-RCNN-FPN-imosauv_strongaug_ema.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
echo "Submitted job1.sh with Job ID: $jobid1"
jobid2=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_ums.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
jobid3=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_ums.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')

jobid4=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_final.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
jobid5=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_final.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')

jobid6=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/ALDI-imosauv_max.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
jobid7=$(sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/MeanTeacher-imosauv_max.yaml "SEED ${seed} LOGGING.GROUP_TAGS S2U,UMS6" | awk '{print $4}')
echo "Submitted job ids $jobid2 $jobid3 $jobid4 $jobid5 $jobid6 $jobid7"
