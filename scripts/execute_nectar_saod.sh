#!/bin/bash

# python tools/train_net.py --config-file configs/urchininf/
# sbatch --partition=accel tools/saga_slurm_train_net.sh 
run="SAOD5"
lr=0.02

python tools/train_net.py --config-file configs/imosauv/OracleWL-RCNN-FPN-redcup_strongaug_ema.yaml "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_full_train_oracle_${lr}/\' LOGGING.GROUP_TAGS ${run},RED,ORACLE SOLVER.BASE_LR ${lr}"
python tools/train_net.py --config-file configs/imosauv/OracleWL-RCNN-FPN-urchin_strongaug_ema.yaml "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_full_train_oracle_${lr}/\' LOGGING.GROUP_TAGS ${run},URCH,ORACLE SOLVER.BASE_LR ${lr}"

coco_file="_og_only"
coco_file_name="OG"

method="CoStudent"
method_name="CoSt_BOTH"

# Annotated bboxes
python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_ft_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},ft_sparse,${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'squidle_urchin_train_sparse\',\)"
python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_ft_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},ft_sparse,${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'squidle_redcup_train_sparse\',\)"

# Clip bboxes
python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17714_train${coco_file}\',\)"
python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17711_train${coco_file}\',\)"

# WEAK Loss
method_name="CoSt_weak"
python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SAOD.STRONG_LOSS 0.0 SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17714_train${coco_file}\',\)"
python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SAOD.STRONG_LOSS 0.0 SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17711_train${coco_file}\',\)"

# STRONG LOSS
method_name="CoSt_str"
python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SAOD.WEAK_LOSS 0.0 SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17714_train${coco_file}\',\)"
python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SAOD.WEAK_LOSS 0.0 SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17711_train${coco_file}\',\)"


coco_file="_cropped_only"
coco_file_name="CROP"

method="CoStudent"
method_name="CoSt"

# Clip bboxes
python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-RCNN-FPN-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/urchin_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17714_train${coco_file}\',\)"
python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} "SEED 1234575 OUTPUT_DIR \'outputs/${run}/redcup_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17711_train${coco_file}\',\)"

