#!/bin/bash

# python tools/train_net.py --config-file configs/urchininf/
# sbatch --partition=accel tools/saga_slurm_train_net.sh 
run="SAOD2"
lr=0.02


coco_file="_noclip"
coco_file_name="NOCLIP"

method="CoStudent"
method_name="CoSt"

#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17630_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17631_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17647_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17648_train${coco_file}\',\)"

method="StudentTeacher"
method_name="ST"

#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17630_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17631_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17647_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17648_train${coco_file}\',\)"



coco_file="_og_only"
coco_file_name="OG"

method="CoStudent"
method_name="CoSt"

#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17630_train${coco_file}\',\)"
sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17631_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17647_train${coco_file}\',\)"
sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17648_train${coco_file}\',\)"

method="StudentTeacher"
method_name="ST"

#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17630_train${coco_file}\',\)"
sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17631_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17647_train${coco_file}\',\)"
sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17648_train${coco_file}\',\)"

coco_file="_cropped_only"
coco_file_name="CROP"

method="CoStudent"
method_name="CoSt"

# URCHIN - 
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17630_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17631_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17647_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17648_train${coco_file}\',\)"

method="StudentTeacher"
method_name="ST"

#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17630_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17631_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17647_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17648_train${coco_file}\',\)"

coco_file=""
coco_file_name="MIX"

method="CoStudent"
method_name="CoSt"

# URCHIN - 
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17630_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17631_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17647_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17648_train${coco_file}\',\)"

method="StudentTeacher"
method_name="ST"

#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17630_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17631_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_many_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,MANY SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17647_train${coco_file}\',\)"
#sbatch --partition=accel tools/saga_slurm_train_net.sh ../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_few_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17648_train${coco_file}\',\)"
