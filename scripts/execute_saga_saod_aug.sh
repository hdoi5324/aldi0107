#!/bin/bash

# python tools/train_net.py --config-file configs/urchininf/
# sbatch --partition=accel scripts/saga_slurm_train_net.sh 
run="SAOD8"
model="RCNN-FPN" # "RCNN-FPN" "FCOS"
model_name="fr" #  "fr" "fcos"
lr=0.02 # 0.02 for FR, 0.01 for FCOS
SEEDS=(3234575) #(1234575 2234575 3234575)
coco_file="_og_only"
coco_file_name="OG"
method="WeakTeacherWStrongStudentW" 
method_name="WTWSSW_i"
priority="iou"  

splits=(50) # (25 50 100 200)
for split in "${splits[@]}"
do
  for SEED in "${SEEDS[@]}"
  do    # Pretrain for 1001 iterations with split only
    #sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/OracleWL-${model}-urchin_strongaug_ema.yaml "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${model_name}_full_train_oracle_${model_name}_${lr}_split${split}/\' SOLVER.MAX_ITER 1001 LOGGING.GROUP_TAGS ${run},URCH,ORACLE SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'squidle_urchin_train_split_${split}\',\)"
    jobid1=$(sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/OracleWL-${model}-redcup_strongaug_ema.yaml \
      "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${model_name}_full_train_oracle_${model_name}_${lr}_split${split}_${SEED}/\' SOLVER.MAX_ITER 1001 LOGGING.GROUP_TAGS ${run},RED,ORACLE SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'squidle_redcup_train_split_${split}\',\)" | awk '{print $4}')
    echo "Submitted job1.sh with Job ID: $jobid1"

    # Sparse training with pretrained model on split and train with split plus sparse
    #sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}_split${split}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,SEMI{$split}_FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17714_train${coco_file}\',\'squidle_urchin_train_split_${split}\',\) MODEL.WEIGHTS \'outputs/${run}/urchin_${model_name}_full_train_oracle_${model_name}_${lr}_split${split}/model_final.pth\'"
    sbatch --dependency=afterok:${jobid1} --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} \
      "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}_split${split}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,SEMI{$split}_FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17711_train${coco_file}\',\'squidle_redcup_train_split_${split}\',\) MODEL.WEIGHTS \'outputs/${run}/redcup_${model_name}_full_train_oracle_${model_name}_${lr}_split${split}_${SEED}/model_final.pth\'"
  done
done

