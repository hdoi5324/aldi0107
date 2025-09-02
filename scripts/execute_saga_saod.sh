#!/bin/bash

# python tools/train_net.py --config-file configs/urchininf/
# sbatch --partition=accel --exclude=c7-7 tools/saga_slurm_train_net.sh 
run="SAOD5"
lr=0.02
SEEDS=(1234575 2234575 3234575)
for SEED in "${SEEDS[@]}"
do
  #sbatch --partition=accel --exclude=c7-7 tools/saga_slurm_train_net.sh ../imosauv/OracleWL-RCNN-FPN-redcup_strongaug_ema.yaml "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_full_train_oracle_${lr}/\' LOGGING.GROUP_TAGS ${run},RED,ORACLE SOLVER.BASE_LR ${lr}"
  #sbatch --partition=accel --exclude=c7-7 tools/saga_slurm_train_net.sh ../imosauv/OracleWL-RCNN-FPN-urchin_strongaug_ema.yaml "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_full_train_oracle_${lr}/\' LOGGING.GROUP_TAGS ${run},URCH,ORACLE SOLVER.BASE_LR ${lr}"
  
  coco_file="_og_only"
  coco_file_name="OG"
  
  method="CoStudent"
  method_name="CoSt_BOTH"
  
  # Annotated bboxes
  #sbatch --partition=accel --exclude=c7-7 tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_ft_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},ft_sparse,${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'squidle_urchin_train_sparse\',\)"
  #sbatch --partition=accel --exclude=c7-7 tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_ft_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},ft_sparse,${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'squidle_redcup_train_sparse\',\)"
  
  # Clip bboxes - our method
  #sbatch --partition=accel --exclude=c7-7 tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17714_train${coco_file}\',\)"
  #sbatch --partition=accel --exclude=c7-7 tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17711_train${coco_file}\',\)"
  
  # WEAK Loss
  method_name="CoSt_weak"
  #sbatch --partition=accel --exclude=c7-7 tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SAOD.STRONG_LOSS 0.0 SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17714_train${coco_file}\',\)"
  #sbatch --partition=accel --exclude=c7-7 tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SAOD.STRONG_LOSS 0.0 SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17711_train${coco_file}\',\)"
  
  # STRONG LOSS
  method_name="CoSt_str"
  #sbatch --partition=accel --exclude=c7-7 tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SAOD.WEAK_LOSS 0.0 SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17714_train${coco_file}\',\)"
  #sbatch --partition=accel --exclude=c7-7 tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SAOD.WEAK_LOSS 0.0 SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17711_train${coco_file}\',\)"
  
  
  coco_file="_cropped_only"
  coco_file_name="CROP"
  
  method="CoStudent"
  method_name="CoSt"
  
  # Clip bboxes
  #sbatch --partition=accel --exclude=c7-7 tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17714_train${coco_file}\',\)"
  #sbatch --partition=accel --exclude=c7-7 tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17711_train${coco_file}\',\)"
  
  
  coco_file="_og_only"
  coco_file_name="OG"
  
  method="CoStudent"
  method_name="CoSt"
  # Clip bboxes - our method - abundant points
  sbatch --partition=accel --exclude=c7-7 tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,MASSFEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17631_train${coco_file}\',\)"
  sbatch --partition=accel --exclude=c7-7 tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,MASSFEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17648_train${coco_file}\',\)"
done