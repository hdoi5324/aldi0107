#!/bin/bash

# python tools/train_net.py --config-file configs/urchininf/
# python tools/train_net.py --config-file configs/urchininf/
run="SAOD16"
models=("RCNN-FPN" "FCOS" "DETR") # "RCNN-FPN" 
model_names=("fr" "fcos" "detr") #  
lrs=(0.02 0.01 0.00005) # 0.02 for FR batch 16 0.01 for FCOS batch 16 check lr for DETR 0.0001??
ims_per_gpu=4
SEEDS=(1234575 2234575 3234575) # 

for i in 1
do 
  model=${models[$i]}
  model_name=${model_names[$i]}
  lr=${lrs[$i]}
  echo ${model} ${model_name} ${lr}
  for SEED in "${SEEDS[@]}"
  do
    # FULLY SUPERVISED ORACLE
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/OracleWL-${model}-urchin_strongaug_ema.yaml SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/urchin_${model_name}_full_train_oracle_${model_name}_${lr}/ LOGGING.GROUP_TAGS ${run},URCH,ORACLE SOLVER.BASE_LR ${lr}
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/OracleWL-${model}-redcup_strongaug_ema.yaml SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/redcup_${model_name}_full_train_oracle_${model_name}_${lr}/ LOGGING.GROUP_TAGS ${run},RED,ORACLE SOLVER.BASE_LR ${lr}

    
    method="WeakTeacherWStrongStudentW" #"CoStudent_bestscore"
    method_name="WTWSSW_i"
    priority="iou"
    
    # SPARSE MANUAL bboxes - sparse (1-2 boxes per image)
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/urchin_${model_name}_ft_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},ft_sparse,${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('squidle_urchin_train_sparse',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/redcup_${model_name}_ft_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},ft_sparse,${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('squidle_redcup_train_sparse',)"
  
  
    method="StrongStudentWWeakStudentS" #"CoStudent_bestscore"
    method_name="SSWWSS_i"
    priority="iou"
    
    # SPARSE MANUAL bboxes - sparse (1-2 boxes per image)
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/urchin_${model_name}_ft_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},ft_sparse,${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('squidle_urchin_train_sparse',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/redcup_${model_name}_ft_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},ft_sparse,${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('squidle_redcup_train_sparse',)"
  
    
    #### POINT BASED BOXES #####
    coco_file="_og_only"
    coco_file_name="OG"
    
    method="WeakTeacherWStrongStudentW" #"CoStudent_bestscore"
    method_name="WTWSSW_s"
    priority="score"
    
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_urchin17714_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_redcup17711_train${coco_file}',)"
    
    method="WeakTeacherWStrongStudentW" 
    method_name="WTWSSW_i"
    priority="iou"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_urchin17714_train${coco_file}',)"
    python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_redcup17711_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-ob_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/ob_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},OB,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_ob17863_train${coco_file}',)"
  
    method="StrongTeacherWWeakStudentW" #"CoStudent_bestscore"
    method_name="STWWSW_s"
    priority="score"
    
    # Clip bboxes - our method
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_urchin17714_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_redcup17711_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-ob_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/ob_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},OB,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_ob17863_train${coco_file}',)"
  
    method_name="STWWSW_i"
    priority="iou"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_urchin17714_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_redcup17711_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-ob_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/ob_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},OB,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_ob17863_train${coco_file}',)"
  
    method="StrongStudentWWeakStudentS" #"CoStudent_bestscore"
    method_name="SSWWSS_s"
    priority="score"
    
    # Clip bboxes - our method
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_urchin17714_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_redcup17711_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-ob_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/ob_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},OB,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_ob17863_train${coco_file}',)"
   
    method="StrongStudentWWeakStudentS" #"CoStudent_bestscore"
    method_name="SSWWSS_i"
    priority="iou"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_urchin17714_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_redcup17711_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-ob_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/ob_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},OB,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_ob17863_train${coco_file}',)"
  
  
    #### END TEST Core combinations
    method="WeakTeacherWStrongStudentW" 
    method_name="WTWSSW_i"
    priority="iou"
    
    # Ablation - strong vs weak loss
    # WEAK Loss
    method_name="WTWSSW_i_weak"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SAOD.STRONG_LOSS 0.0 SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_urchin17714_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SAOD.STRONG_LOSS 0.0 SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_redcup17711_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-ob_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/ob_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},OB,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_ob17863_train${coco_file}',)"
    
    # STRONG LOSS
    method_name="WTWSSW_i_str"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SAOD.WEAK_LOSS 0.0 SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_urchin17714_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} "SAOD.WEAK_LOSS 0.0 SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_redcup17711_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-ob_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/ob_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},OB,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_ob17863_train${coco_file}',)"
    
    ##### CROPPED ######
    coco_file="_cropped_only"
    coco_file_name="CROP"
    
    method="WeakTeacherWStrongStudentW" 
    method_name="WTWSSW_i"
    priority="iou"
    
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_urchin17714_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_redcup17711_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-ob_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/ob_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},OB,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_ob17863_train${coco_file}',)"
  
    method="StrongStudentWWeakStudentS" #"CoStudent_bestscore"
    method_name="SSWWSS_i"
    priority="iou"
    
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_urchin17714_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_redcup17711_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-ob_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/ob_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},OB,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_ob17863_train${coco_file}',)"
    
    ##### END CROPPED #####
  
    coco_file="_og_only"
    coco_file_name="OG"
    
    method="WTWSSW"
    method_name="WTWSSW_i"
    # From weak points - mass few
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,MASSFEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_urchin17631_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,MASSFEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_redcup17648_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-ob_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/ob_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},OB,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_ob17863_train${coco_file}',)"
  
    coco_file="_cropped_only"
    coco_file_name="CROP"
    # From weak points - mass few cropped
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-urchin_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/urchin_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},URCH,MASSFEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_urchin17631_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-redcup_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/redcup_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},RED,MASSFEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_redcup17648_train${coco_file}',)"
    #python tools/train_net.py --config-file configs/urchininf/../imosauv/CoStudent-${model}-ob_strongaug_ema.yaml SAOD.LABELING_METHOD ${method} SAOD.DENOISE_PRIORITY ${priority} SEED ${SEED} SOLVER.IMS_PER_GPU ${ims_per_gpu} OUTPUT_DIR outputs/${run}/ob_${model_name}_${coco_file_name}_sparse_${method_name}_${lr}/ LOGGING.GROUP_TAGS ${run},${coco_file_name},${method_name},OB,FEW SOLVER.BASE_LR ${lr} DATASETS.TRAIN "('loose_ob17863_train${coco_file}',)"
  
  done
done