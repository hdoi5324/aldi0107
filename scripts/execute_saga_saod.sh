#!/bin/bash

# python tools/train_net.py --config-file configs/urchininf/
# sbatch scripts/saga_slurm_train_net_detr.sh 
run="SAOD9"
model="DETR" # "RCNN-FPN" "FCOS"
model_name="detr" #  "fr" "fcos"
lr=0.0001 # 0.02 for FR 0.01 for FCOS
SEEDS=(1234575 2234575 3234575)
for SEED in "${SEEDS[@]}"
do
  # Oracle
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/OracleWL-${model}-redcup_strongaug_ema.yaml "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${model_name}_full_train_oracle_${model_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},RED,ORACLE SOLVER.BASE_LR ${lr}"
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/OracleWL-${model}-urchin_strongaug_ema.yaml "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${model_name}_full_train_oracle_${model_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},URCH,ORACLE SOLVER.BASE_LR ${lr}"
  
  coco_file="_og_only"
  coco_file_name="OG"
  
  method="WeakTeacherWStrongStudentW" #"CoStudent_bestscore"
  method_name="WTWSSW_i"
  priority="iou"
  
  # Manually annotated bboxes - all and sparse
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${model_name}_ft_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},ft_sparse,${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'squidle_urchin_train_sparse\',\)"
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${model_name}_ft_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},ft_sparse,${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'squidle_redcup_train_sparse\',\)"
  
  
  #### TEST Core combinations
  method="WeakTeacherWStrongStudentW" #"CoStudent_bestscore"
  method_name="WTWSSW_s"
  priority="score"
  
  # Clip bboxes - our method
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17714_train${coco_file}\',\)"
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17711_train${coco_file}\',\)"
  
  method_name="WTWSSW_i"
  priority="iou"
  sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17714_train${coco_file}\',\)"
  sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17711_train${coco_file}\',\)"

  #method="StrongTeacherWWeakStudentW" #"CoStudent_bestscore"
  #method_name="STWWSW_s"
  #priority="score"
  
  # Clip bboxes - our method
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17714_train${coco_file}\',\)"
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17711_train${coco_file}\',\)"
  
  #method_name="STWWSW_i"
  #priority="iou"
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17714_train${coco_file}\',\)"
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17711_train${coco_file}\',\)"

  method="StrongStudentWWeakStudentS" #"CoStudent_bestscore"
  #method_name="SSWWSS_s"
  #priority="score"
  
  # Clip bboxes - our method
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17714_train${coco_file}\',\)"
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17711_train${coco_file}\',\)"
  
  method="StrongStudentWWeakStudentS" #"CoStudent_bestscore"
  method_name="SSWWSS_i"
  priority="iou"
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17714_train${coco_file}\',\)"
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17711_train${coco_file}\',\)"


  #### END TEST Core combinations
  method_name="WTWSSW_i"
  priority="iou"
  
  # Ablation - strong vs weak loss
  # WEAK Loss
  method_name="WTWSSW_i_weak"
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SAOD.STRONG_LOSS 0.0 SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17714_train${coco_file}\',\)"
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SAOD.STRONG_LOSS 0.0 SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17711_train${coco_file}\',\)"
  
  # STRONG LOSS
  method_name="WTWSSW_i_str"
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SAOD.WEAK_LOSS 0.0 SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17714_train${coco_file}\',\)"
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SAOD.WEAK_LOSS 0.0 SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17711_train${coco_file}\',\)"
  
  
  coco_file="_cropped_only"
  coco_file_name="CROP"
  
  method="WTWSSW"
  method_name="WTWSSW_i"
  
  # Clip bboxes
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17714_train${coco_file}\',\)"
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17711_train${coco_file}\',\)"
  
  
  coco_file="_og_only"
  coco_file_name="OG"
  
  method="WTWSSW"
  method_name="WTWSSW_i"
  # From weak points - mass few
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,MASSFEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17631_train${coco_file}\',\)"
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,MASSFEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17648_train${coco_file}\',\)"

  coco_file="_cropped_only"
  coco_file_name="CROP"
  # From weak points - mass few cropped
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,MASSFEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_urchin17631_train${coco_file}\',\)"
  #sbatch scripts/saga_slurm_train_net_detr.sh ../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SEED ${SEED} OUTPUT_DIR \'outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/\' LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,MASSFEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN \(\'loose_redcup17648_train${coco_file}\',\)"

done