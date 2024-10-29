#!/bin/bash

#sbatch  --partition=accel tools/saga_slurm_train_net.sh


# Base Train Inf Test SQ
sbatch tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_weakaug.yaml "OUTPUT_DIR \'outputs/urchininf/base_weakaug_sq/\' DATASETS.TEST \(\'squidle_urchin_2011_test\',\)  LOGGING.GROUP_TAGS Inf2SQ,T3 "
sbatch tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema.yaml  "OUTPUT_DIR \'outputs/urchininf/base_strongaug_ema_sq/\' DATASETS.TEST \(\'squidle_urchin_2011_test\',\)  LOGGING.GROUP_TAGS Inf2SQ,T3 "

# Base synthetic all classes
python tools/train_net.py --config-file configs/urchininf/Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/base_strongaug_ema_inf_all/\' LOGGING.GROUP_TAGS InfAll2SQ,T3MODEL.ROI_HEADS.NUM_CLASSES 8 DATASETS.TRAIN \(\'urchininf_v0_train_all\',\'urchininf_rov_v1_train_all\',\'urchininf_auv_v2_train_all\',\) DATASETS.TEST \(\'urchininf_v0_test_all\',\'urchininf_rov_v1_test_all\',\'urchininf_auv_v2_test_all\',\)

# Base real all classes
python tools/train_net.py --config-file configs/urchininf/Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/urchininf_base_strongaug_ema_sq_all/\' LOGGING.GROUP_TAGS SqAll,T3 MODEL.ROI_HEADS.NUM_CLASSES 6 DATASETS.TRAIN \(\'squidle_pretrain_train\',\) DATASETS.TEST \(\'squidle_pretrain_test\',\)

# Base UDD training 
sbatch tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_weakaug.yaml "DATASETS.TEST \(\'UDD_test\',\)  LOGGING.GROUP_TAGS Inf2UDD,T3 OUTPUT_DIR \'outputs/urchininf/urchininf_base_weakaug_udd/\'"
sbatch tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema.yaml "DATASETS.TEST \(\'UDD_test\',\)  LOGGING.GROUP_TAGS Inf2UDD,T3 OUTPUT_DIR \'outputs/urchininf/urchininf_base_strongaug_ema_udd/\'"
sbatch tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema.yaml "DATASETS.TEST \(\'UDD_test\',\)  LOGGING.GROUP_TAGS Inf2UDD,PAll,T3 OUTPUT_DIR \'outputs/urchininf/urchininf_base_strongaug_ema_udd_all/\' MODEL.ROI_HEADS.NUM_CLASSES 8 DATASETS.TRAIN \(\'urchininf_v0_train_all\',\'urchininf_rov_v1_train_all\',\'urchininf_auv_v2_train_all\',\)"

# Eval only
sbatch tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_weakaug.yaml --eval-only "MODEL.WEIGHTS \'outputs/urchininf/urchininf_base_weakaug/model_final.pth\' DATASETS.TEST \(\'UDD_test\',\)  LOGGING.GROUP_TAGS Inf2UDD,T3 "
sbatch tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema.yaml --eval-only "MODEL.WEIGHTS \'outputs/urchininf/urchininf_base_strongaug_ema/model_final.pth\' DATASETS.TEST \(\'UDD_test\',\)  LOGGING.GROUP_TAGS Inf2UDD,T3 "

sbatch tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_weakaug.yaml  "DATASETS.TEST \(\'UDD_test\',\)  LOGGING.GROUP_TAGS Inf2UDD,T3 "
sbatch tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema.yaml  "DATASETS.TEST \(\'UDD_test\',\)  LOGGING.GROUP_TAGS Inf2UDD,T3 "



# Best model strongaugEMA
sbatch tools/saga_slurm_train_net.sh ALDI-urchininf.yaml "DATASETS.TEST \(\'squidle_urchin_2011_test\',\) DATASETS.UNLABELED \(\'squidle_urchin_2009_train\',\'squidle_east_tas_urchins_train\',\) LOGGING.GROUP_TAGS Inf2SQ,T3,BestP MODEL.WEIGHTS \'outputs/urchininf/urchininf_base_strongaug_ema/squidle_urchin_2011_test_model_best.pth\'"
sbatch tools/saga_slurm_train_net.sh ALDI-urchininf.yaml "DATASETS.TEST \(\'UDD_test\',\) DATASETS.UNLABELED \(\'UDD_train\',\) LOGGING.GROUP_TAGS Inf2UDD,T3,BestP  MODEL.WEIGHTS \'outputs/urchininf/urchininf_base_strongaug_ema_udd/udd_test_model_best.pth\'"

# DAOD Pre-training ALL classes
python tools/train_net.py --config-file configs/urchininf/ALDI-urchininf.yaml "DATASETS.TEST \(\'squidle_urchin_2011_test\',\) DATASETS.UNLABELED \(\'squidle_urchin_2009_train\',\'squidle_east_tas_urchins_train\',\) LOGGING.GROUP_TAGS Inf2SQ,T3,AllRealP MODEL.WEIGHTS \'outputs/urchininf/urchininf_base_strongaug_ema_sq_all/squidle_pretrain_test_model_best.pth\'
python tools/train_net.py --config-file configs/urchininf/ALDI-urchininf.yaml "DATASETS.TEST \(\'UDD_test\',\) DATASETS.UNLABELED \(\'UDD_train\',\) LOGGING.GROUP_TAGS Inf2UDD,T3,AllRealP  MODEL.WEIGHTS \'outputs/urchininf/urchininf_base_strongaug_ema_sq_all/squidle_pretrain_test_model_best.pth\'

# DAOD Pre-training ALL classes - real images
python tools/train_net.py --config-file configs/urchininf/ALDI-urchininf.yaml "DATASETS.TEST \(\'squidle_urchin_2011_test\',\) DATASETS.UNLABELED \(\'squidle_urchin_2009_train\',\'squidle_east_tas_urchins_train\',\) LOGGING.GROUP_TAGS Inf2SQ,T3,AllP MODEL.WEIGHTS \'outputs/urchininf/urchininf_base_strongaug_ema_sq_all/model_final.pth\'
python tools/train_net.py --config-file configs/urchininf/ALDI-urchininf.yaml "DATASETS.TEST \(\'UDD_test\',\) DATASETS.UNLABELED \(\'UDD_train\',\) LOGGING.GROUP_TAGS Inf2UDD,T3,AllP  MODEL.WEIGHTS \'outputs/urchininf/urchininf_base_strongaug_ema_sq_all/model_final.pth\'


#sbatch tools/saga_slurm_train_net.sh urchininf_priorart/SADA_urchininf.yaml "DATASETS.TEST \(\'squidle_urchin_2011_test\',\) DATASETS.UNLABELED \(\'squidle_urchin_2009_train\',\'squidle_east_tas_urchins_train\',\) LOGGING.GROUP_TAGS Inf2SQ,T3  SOLVER.IMS_PER_GPU 4 SOLVER.BASE_LR 0.003
#sbatch tools/saga_slurm_train_net.sh urchininf_priorart/MIC-urchininf_NoBurnIn.yaml "DATASETS.TEST \(\'squidle_urchin_2011_test\',\) DATASETS.UNLABELED \(\'squidle_urchin_2009_train\',\'squidle_east_tas_urchins_train\',\) LOGGING.GROUP_TAGS Inf2SQ,T3  SOLVER.BASE_LR 0.003


# Oracle
sbatch tools/saga_slurm_train_net.sh OracleT-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/oracle_strongaug_ema_sq/\' DATASETS.TEST \(\'squidle_urchin_2011_test\',\) DATASETS.TRAIN \(\'squidle_urchin_2009_train\',\'squidle_east_tas_urchins_train\',\) LOGGING.GROUP_TAGS Sq,T3"
sbatch tools/saga_slurm_train_net.sh OracleT-RCNN-FPN-urchininf_weakaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/oracle_weakaug_ema_sq/\' DATASETS.TEST \(\'squidle_urchin_2011_test\',\) DATASETS.TRAIN \(\'squidle_urchin_2009_train\',\'squidle_east_tas_urchins_train\',\) LOGGING.GROUP_TAGS Sq,T3"
sbatch tools/saga_slurm_train_net.sh OracleT-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/oracle_strongaug_ema_udd/\' DATASETS.TEST \(\'UDD_test\',\) DATASETS.TRAIN \(\'UDD_train\',\) LOGGING.GROUP_TAGS UDD,T3"
sbatch tools/saga_slurm_train_net.sh OracleT-RCNN-FPN-urchininf_weakaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/oracle_weakaug_ema_udd/\' DATASETS.TEST \(\'UDD_test\',\) DATASETS.TRAIN \(\'UDD_train\',\'squidle_east_tas_urchins_train\',\) LOGGING.GROUP_TAGS UDD,T3"



# Unlabelled without instances
# DAOD
sbatch --partition=accel tools/saga_slurm_train_net.sh MeanTeacher-urchininf.yaml "DATASETS.TEST \(\'squidle_urchin_2011_test\',\) DATASETS.UNLABELED \(\'squidle_urchin_2009_train_without_target\',\'squidle_east_tas_urchins_train_without_target\',\) LOGGING.GROUP_TAGS Inf2SQ,T3,NoTgt "
sbatch --partition=accel tools/saga_slurm_train_net.sh ALDI-urchininf.yaml "DATASETS.TEST \(\'squidle_urchin_2011_test\',\) DATASETS.UNLABELED \(\'squidle_urchin_2009_train_without_target\',\'squidle_east_tas_urchins_train_without_target\',\) LOGGING.GROUP_TAGS Inf2SQ,T3,NoTgt"
sbatch --partition=accel tools/saga_slurm_train_net.sh urchininf_priorart/AT-urchininf.yaml "DATASETS.TEST \(\'squidle_urchin_2011_test\',\) DATASETS.UNLABELED \(\'squidle_urchin_2009_train_without_target\',\'squidle_east_tas_urchins_train_without_target\',\) LOGGING.GROUP_TAGS Inf2SQ,T3,NoTgt"

sbatch --partition=accel tools/saga_slurm_train_net.sh urchininf_priorart/SADA_urchininf.yaml "DATASETS.TEST \(\'squidle_urchin_2011_test\',\) DATASETS.UNLABELED \(\'squidle_urchin_2009_train_without_target\',\'squidle_east_tas_urchins_train_without_target\',\) LOGGING.GROUP_TAGS Inf2SQ,T3,NoTgt "

#### BASE
# Inf With some Squidle target labeled data - Inf+no sq, 200, 400, 600, All sq+Inf
python tools/train_net.py --config-file Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/base_strongaug_ema_inf_sq/\' DATASETS.TEST \(\'squidle_urchin_2011_test\',\)  LOGGING.GROUP_TAGS Inf2SQ,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\)"
python tools/train_net.py --config-file Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/base_inf_sq200/\' LOGGING.GROUP_TAGS Inf+200tgt2SQ,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\) "
python tools/train_net.py --config-file Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/base_inf_sq400/\' LOGGING.GROUP_TAGS Inf+400tgt2SQ,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\'squidle_urchin_2009_train_split1\',\) "
python tools/train_net.py --config-file Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/base_inf_sq600/\' LOGGING.GROUP_TAGS Inf+600tgt2SQ,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\'squidle_urchin_2009_train_split1\',\'squidle_urchin_2009_train_split2\',\) "
python tools/train_net.py --config-file Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/base_inf_sqall/\' LOGGING.GROUP_TAGS Inf+alltgt2SQ,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train\',\'squidle_east_tas_urchins_train\',\) "
python tools/train_net.py --config-file Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/base_inf_sq100/\' LOGGING.GROUP_TAGS Inf+100tgt2SQ,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split100\',\) "

# Inf With some UDD target labeled data - Inf+no sq, 200, 400, 600, All UDD+Inf
python tools/train_net.py --config-file Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/base_strongaug_ema_inf_udd/\' DATASETS.TEST \(\'UDD_test\',\)  LOGGING.GROUP_TAGS Inf2UDD,T3 "
python tools/train_net.py --config-file Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/base_strongaug_ema_inf_udd200/\' LOGGING.GROUP_TAGS Inf+200tgt2UDD,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\) "
python tools/train_net.py --config-file Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/base_strongaug_ema_inf_udd400/\' LOGGING.GROUP_TAGS Inf+400tgt2UDD,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\'UDD_train_split1\',\) "
python tools/train_net.py --config-file Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/base_strongaug_ema_inf_udd600/\' LOGGING.GROUP_TAGS Inf+600tgt2UDD,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\'UDD_train_split1\',\'UDD_train_split2\',\) "
python tools/train_net.py --config-file Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/base_strongaug_ema_inf_uddall/\' LOGGING.GROUP_TAGS Inf+alltgt2UDD,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train\',\) "
python tools/train_net.py --config-file Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/base_strongaug_ema_inf_udd100/\' LOGGING.GROUP_TAGS Inf+100tgt2UDD,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split100\',\) "

#### DAOD
### WITH STRONGAUG PRETRAIN
# MeanTeacher Inf With some Squidle target labeled data - Inf+no sq, 200, 400, 600, All sq+Inf
python tools/train_net.py --config-file MeanTeacher-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_sq/\' LOGGING.GROUP_TAGS Inf2SQ,T3"
sbatch  --partition=accel  tools/saga_slurm_train_net.sh MeanTeacher-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_sq200/\' LOGGING.GROUP_TAGS Inf+200tgt2SQ,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\) "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh MeanTeacher-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_sq400/\' LOGGING.GROUP_TAGS Inf+400tgt2SQ,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\'squidle_urchin_2009_train_split1\',\) "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh MeanTeacher-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_sq600/\' LOGGING.GROUP_TAGS Inf+600tgt2SQ,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\'squidle_urchin_2009_train_split1\',\'squidle_urchin_2009_train_split2\',\) "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh MeanTeacher-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_sqall/\' LOGGING.GROUP_TAGS Inf+alltgt2SQ,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train\',\'squidle_east_tas_urchins_train\',\) "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh MeanTeacher-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_sq100/\' LOGGING.GROUP_TAGS Inf+100tgt2SQ,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split100\',\) "

# MeanTeacher Inf with some UDD labeled data 
sbatch  --partition=accel  tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_udd/\' LOGGING.GROUP_TAGS Inf2UDD,T3 "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh MeanTeacher-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_udd200/\' LOGGING.GROUP_TAGS Inf+200tgt2UDD,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\) "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh MeanTeacher-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_udd400/\' LOGGING.GROUP_TAGS Inf+400tgt2UDD,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\'UDD_train_split1\',\) "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh MeanTeacher-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_udd600/\' LOGGING.GROUP_TAGS Inf+600tgt2UDD,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\'UDD_train_split1\',\'UDD_train_split2\',\) "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh MeanTeacher-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_uddall/\' LOGGING.GROUP_TAGS Inf+alltgt2UDD,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train\',\) "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh MeanTeacher-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_udd100/\' LOGGING.GROUP_TAGS Inf+100tgt2UDD,T3  DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split100\',\) "

# ALDI Inf With some Squidle target labeled data - Inf+no sq, 200, 400, 600, All sq+Inf
python tools/train_net.py --config-file ALDI-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_sq/\' LOGGING.GROUP_TAGS Inf2SQ,T3 "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh ALDI-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_sq200/\' LOGGING.GROUP_TAGS Inf+200tgt2SQ,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\) "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh ALDI-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_sq400/\' LOGGING.GROUP_TAGS Inf+400tgt2SQ,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\'squidle_urchin_2009_train_split1\',\) "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh ALDI-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_sq600/\' LOGGING.GROUP_TAGS Inf+600tgt2SQ,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\'squidle_urchin_2009_train_split1\',\'squidle_urchin_2009_train_split2\',\) "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh ALDI-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_sqall/\' LOGGING.GROUP_TAGS Inf+alltgt2SQ,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train\',\'squidle_east_tas_urchins_train\',\) "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh ALDI-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_sq100/\' LOGGING.GROUP_TAGS Inf+100tgt2SQ,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split100\',\) "

# ALDI Inf with some UDD labeled data -
sbatch  --partition=accel  tools/saga_slurm_train_net.sh ALDI-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_udd/\' LOGGING.GROUP_TAGS Inf2UDD,T3"
sbatch  --partition=accel  tools/saga_slurm_train_net.sh ALDI-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_udd200/\' LOGGING.GROUP_TAGS Inf+200tgt2UDD,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\) "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh ALDI-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_udd400/\' LOGGING.GROUP_TAGS Inf+400tgt2UDD,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\'UDD_train_split1\',\) "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh ALDI-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_udd600/\' LOGGING.GROUP_TAGS Inf+600tgt2UDD,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\'UDD_train_split1\',\'UDD_train_split2\',\) "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh ALDI-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_uddall/\' LOGGING.GROUP_TAGS Inf+alltgt2UDD,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train\',\) "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh ALDI-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_udd100/\' LOGGING.GROUP_TAGS Inf+100tgt2UDD,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split100\',\) "

# AT Inf With some Squidle target labeled data - Inf+no sq, 200, 400, 600, All sq+Inf
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --config-file configs/urchininf/urchininf_priorart/AT-urchininf_sq.yaml OUTPUT_DIR \'outputs/urchininf/at_inf_sq/\' LOGGING.GROUP_TAGS Inf2SQ,T3
CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py --num-gpus 2 --config-file configs/urchininf/urchininf_priorart/AT-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_sq200/\' LOGGING.GROUP_TAGS Inf+200tgt2SQ,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\) "
CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py --num-gpus 2 --config-file configs/urchininf/urchininf_priorart/AT-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_sq400/\' LOGGING.GROUP_TAGS Inf+400tgt2SQ,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\'squidle_urchin_2009_train_split1\',\) "
CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py --num-gpus 2 --config-file configs/urchininf/urchininf_priorart/AT-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_sq600/\' LOGGING.GROUP_TAGS Inf+600tgt2SQ,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\'squidle_urchin_2009_train_split1\',\'squidle_urchin_2009_train_split2\',\) "
CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py --num-gpus 2 --config-file configs/urchininf/urchininf_priorart/AT-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_sqall/\' LOGGING.GROUP_TAGS Inf+alltgt2SQ,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train\',\'squidle_east_tas_urchins_train\',\) "
CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py --num-gpus 2 --config-file configs/urchininf/urchininf_priorart/AT-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_sq100/\' LOGGING.GROUP_TAGS Inf+100tgt2SQ,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split100\',\) "


# AT Inf with some UDD labeled data 
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --config-file configs/urchininf/urchininf_priorart/AT-urchininf.yaml OUTPUT_DIR \'outputs/urchininf/at_inf_udd/\' LOGGING.GROUP_TAGS Inf2UDD,T3 && sudo shutdown
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --config-file configs/urchininf/urchininf_priorart/AT-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_udd200/\' LOGGING.GROUP_TAGS Inf+200tgt2UDD,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\) "
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --config-file configs/urchininf/urchininf_priorart/AT-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_udd400/\' LOGGING.GROUP_TAGS Inf+400tgt2UDD,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\'UDD_train_split1\',\) "
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --config-file configs/urchininf/urchininf_priorart/AT-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_udd600/\' LOGGING.GROUP_TAGS Inf+600tgt2UDD,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\'UDD_train_split1\',\'UDD_train_split2\',\) "
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --config-file configs/urchininf/urchininf_priorart/AT-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_uddall/\' LOGGING.GROUP_TAGS Inf+alltgt2UDD,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train\',\) "
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --config-file configs/urchininf/urchininf_priorart/AT-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_udd100/\' LOGGING.GROUP_TAGS Inf+100tgt2UDD,T3 DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split100\',\) "

CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --config-file configs/urchininf/urchininf_priorart/SADA_urchininf_sq.yaml OUTPUT_DIR \'outputs/urchininf/sada_inf_sq/\' LOGGING.GROUP_TAGS Inf2SQ,T3
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --config-file configs/urchininf/urchininf_priorart/SADA_urchininf.yaml OUTPUT_DIR \'outputs/urchininf/sada_inf_udd/\' LOGGING.GROUP_TAGS Inf2UDD,T3 && sudo shutdown

sbatch  --partition=accel  tools/saga_slurm_train_net.sh urchininf_priorart/SADA_urchininf_sq.yaml OUTPUT_DIR \'outputs/urchininf/sada_inf_sq/\' LOGGING.GROUP_TAGS Inf2SQ,T3
sbatch  --partition=accel  tools/saga_slurm_train_net.sh urchininf_priorart/SADA_urchininf.yaml OUTPUT_DIR \'outputs/urchininf/sada_inf_udd/\' LOGGING.GROUP_TAGS Inf2UDD,T3 && sudo shutdown

CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py --num-gpus 2 --config-file configs/urchininf/urchininf_priorart/MIC-urchininf_sq.yaml OUTPUT_DIR \'outputs/urchininf/mic_inf_sq/\' LOGGING.GROUP_TAGS Inf2SQ,T3
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py  --dist-url 'tcp://127.0.0.1:50162' --num-gpus 2 --config-file configs/urchininf/urchininf_priorart/MIC-urchininf.yaml OUTPUT_DIR \'outputs/urchininf/mic_inf_udd/\' LOGGING.GROUP_TAGS Inf2UDD,T3 && sudo shutdown


### NO PRETRAINING 
# MeanTeacher Inf With some Squidle target labeled data - Inf+no sq, 200, 400, 600, All sq+Inf
python tools/train_net.py --config-file MeanTeacher-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_nopt/\' LOGGING.GROUP_TAGS Inf2SQ,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\) "
python tools/train_net.py --config-file MeanTeacher-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_sq200_nopt/\' LOGGING.GROUP_TAGS Inf+200tgt2SQ,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\) "
python tools/train_net.py --config-file MeanTeacher-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_sq400_nopt/\' LOGGING.GROUP_TAGS Inf+400tgt2SQ,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\'squidle_urchin_2009_train_split1\',\) "
python tools/train_net.py --config-file MeanTeacher-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_sq600_nopt/\' LOGGING.GROUP_TAGS Inf+600tgt2SQ,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\'squidle_urchin_2009_train_split1\',\'squidle_urchin_2009_train_split2\',\) "
python tools/train_net.py --config-file MeanTeacher-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_sqall_nopt/\' LOGGING.GROUP_TAGS Inf+alltgt2SQ,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train\',\'squidle_east_tas_urchins_train\',\) "
python tools/train_net.py --config-file MeanTeacher-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_sq100_nopt/\' LOGGING.GROUP_TAGS Inf+100tgt2SQ,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split100\',\) "

# MeanTeacher Inf with some UDD labeled data - no pretrain
python tools/train_net.py --config-file MeanTeacher-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_nopt/\' LOGGING.GROUP_TAGS Inf2UDD,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\) "
python tools/train_net.py --config-file MeanTeacher-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_udd200_nopt/\' LOGGING.GROUP_TAGS Inf+200tgt2UDD,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\) "
python tools/train_net.py --config-file MeanTeacher-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_udd400_nopt/\' LOGGING.GROUP_TAGS Inf+400tgt2UDD,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\'UDD_train_split1\',\) "
python tools/train_net.py --config-file MeanTeacher-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_udd600_nopt/\' LOGGING.GROUP_TAGS Inf+600tgt2UDD,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\'UDD_train_split1\',\'UDD_train_split2\',\) "
python tools/train_net.py --config-file MeanTeacher-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_uddall_nopt/\' LOGGING.GROUP_TAGS Inf+alltgt2UDD,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train\',\) "
python tools/train_net.py --config-file MeanTeacher-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/mt_inf_udd100_nopt/\' LOGGING.GROUP_TAGS Inf+100tgt2UDD,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split100\',\) "

# ALDI Inf With some Squidle target labeled data - Inf+no sq, 200, 400, 600, All sq+Inf
python tools/train_net.py --config-file ALDI-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_nopt/\' LOGGING.GROUP_TAGS Inf2SQ,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\) "
python tools/train_net.py --config-file ALDI-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_sq200_nopt/\' LOGGING.GROUP_TAGS Inf+200tgt2SQ,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\) "
python tools/train_net.py --config-file ALDI-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_sq400_nopt/\' LOGGING.GROUP_TAGS Inf+400tgt2SQ,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\'squidle_urchin_2009_train_split1\',\) "
python tools/train_net.py --config-file ALDI-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_sq600_nopt/\' LOGGING.GROUP_TAGS Inf+600tgt2SQ,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\'squidle_urchin_2009_train_split1\',\'squidle_urchin_2009_train_split2\',\) "
python tools/train_net.py --config-file ALDI-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_sqall_nopt/\' LOGGING.GROUP_TAGS Inf+alltgt2SQ,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train\',\'squidle_east_tas_urchins_train\',\) "
python tools/train_net.py --config-file ALDI-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_sq100_nopt/\' LOGGING.GROUP_TAGS Inf+100tgt2SQ,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split100\',\) "

# ALDI Inf with some UDD labeled data - no pretrain
python tools/train_net.py --config-file ALDI-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_nopt/\' LOGGING.GROUP_TAGS Inf2UDD,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\) "
python tools/train_net.py --config-file ALDI-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_udd200_nopt/\' LOGGING.GROUP_TAGS Inf+200tgt2UDD,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\) "
python tools/train_net.py --config-file ALDI-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_udd400_nopt/\' LOGGING.GROUP_TAGS Inf+400tgt2UDD,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\'UDD_train_split1\',\) "
python tools/train_net.py --config-file ALDI-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_udd600_nopt/\' LOGGING.GROUP_TAGS Inf+600tgt2UDD,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\'UDD_train_split1\',\'UDD_train_split2\',\) "
python tools/train_net.py --config-file ALDI-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_uddall_nopt/\' LOGGING.GROUP_TAGS Inf+alltgt2UDD,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train\',\) "
python tools/train_net.py --config-file ALDI-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/aldi_inf_udd100_nopt/\' LOGGING.GROUP_TAGS Inf+100tgt2UDD,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split100\',\) "

# SADA Inf With some Squidle target labeled data - Inf+no sq, 200, 400, 600, All sq+Inf
python tools/train_net.py --config-file urchininf_priorart/AT-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_nopt/\' LOGGING.GROUP_TAGS Inf2SQ,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\) "
python tools/train_net.py --config-file urchininf_priorart/AT-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_sq200_nopt/\' LOGGING.GROUP_TAGS Inf+200tgt2SQ,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\) "
python tools/train_net.py --config-file urchininf_priorart/AT-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_sq400_nopt/\' LOGGING.GROUP_TAGS Inf+400tgt2SQ,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\'squidle_urchin_2009_train_split1\',\) "
python tools/train_net.py --config-file urchininf_priorart/AT-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_sq600_nopt/\' LOGGING.GROUP_TAGS Inf+600tgt2SQ,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split0\',\'squidle_urchin_2009_train_split1\',\'squidle_urchin_2009_train_split2\',\) "
python tools/train_net.py --config-file urchininf_priorart/AT-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_sqall_nopt/\' LOGGING.GROUP_TAGS Inf+alltgt2SQ,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train\',\'squidle_east_tas_urchins_train\',\) "
python tools/train_net.py --config-file urchininf_priorart/AT-urchininf_sq.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_sq100_nopt/\' LOGGING.GROUP_TAGS Inf+100tgt2SQ,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'squidle_urchin_2009_train_split100\',\) "


# SADA Inf with some UDD labeled data - no pretrain
python tools/train_net.py --config-file urchininf_priorart/AT-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_nopt/\' LOGGING.GROUP_TAGS Inf2UDD,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\) "
python tools/train_net.py --config-file urchininf_priorart/AT-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_udd200_nopt/\' LOGGING.GROUP_TAGS Inf+200tgt2UDD,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\) "
python tools/train_net.py --config-file urchininf_priorart/AT-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_udd400_nopt/\' LOGGING.GROUP_TAGS Inf+400tgt2UDD,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\'UDD_train_split1\',\) "
python tools/train_net.py --config-file urchininf_priorart/AT-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_udd600_nopt/\' LOGGING.GROUP_TAGS Inf+600tgt2UDD,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split0\',\'UDD_train_split1\',\'UDD_train_split2\',\) "
python tools/train_net.py --config-file urchininf_priorart/AT-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_uddall_nopt/\' LOGGING.GROUP_TAGS Inf+alltgt2UDD,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train\',\) "
python tools/train_net.py --config-file urchininf_priorart/AT-urchininf.yaml "OUTPUT_DIR \'outputs/urchininf/at_inf_udd100_nopt/\' LOGGING.GROUP_TAGS Inf+100tgt2UDD,T3,NoP MODEL.WEIGHTS \'models/model_final_f10217.pkl\' DATASETS.TRAIN \(\'urchininf_v0_train\',\'urchininf_rov_v1_train\',\'urchininf_auv_v2_train\',\'UDD_train_split100\',\) "
