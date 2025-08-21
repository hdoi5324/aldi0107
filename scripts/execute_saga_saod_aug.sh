#!/bin/bash

# python tools/train_net.py --config-file configs/urchininf/
# sbatch --partition=accel scripts/saga_slurm_train_net.sh 
lr=0.02
seed=2234575 #1234575

coco_file="_og_only"
coco_file_name="OG"

method="CoStudent"
method_name="CoSt"

splits=(25 50 100 200)
for split in "${splits[@]}"
do
  run="SAOD3_${split}"
  #sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED ${seed} OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17630_train${coco_file}\',\'squidle_urchin_train_split_${split}\',\)"
  #sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED ${seed} OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17631_train${coco_file}\',\'squidle_urchin_train_split_${split}\',\)"
  #sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED ${seed} OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17647_train${coco_file}\',\'squidle_redcup_train_split_${split}\',\)"
  sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED ${seed} OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17648_train${coco_file}\',\'squidle_redcup_train_split_${split}\',\)"
  
  method="StudentTeacher"
  method_name="ST"
  
  #sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED ${seed} OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17630_train${coco_file}\',\'squidle_urchin_train_split_${split}\',\)"
  #sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED ${seed} OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17631_train${coco_file}\',\'squidle_urchin_train_split_${split}\',\)"
  #sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED ${seed} OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17647_train${coco_file}\',\'squidle_redcup_train_split_${split}\',\)"
  sbatch --partition=accel scripts/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED ${seed} OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17648_train${coco_file}\',\'squidle_redcup_train_split_${split}\',\)"
done

