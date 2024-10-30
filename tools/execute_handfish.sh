#!/bin/bash

# python tools/train_net.py --config-file configs/urchininf/
# sbatch --partition=accel tools/saga_slurm_train_net.sh 
python tools/train_net.py --eval-only --config-file configs/urchininf/OracleT-RCNN-FPN-urchininf_weakaug.yaml MODEL.WEIGHTS outputs/handfishinf/oracle_weakaug_sq/model_0011999.pth DATASETS.TEST \(\'squidle_handfish_15800_test\',\)


# Base Train  - Weak Strong
sbatch --partition=accel tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_weakaug.yaml "OUTPUT_DIR \'outputs/handfishinf/base_weakaug_sq/\' DATASETS.TRAIN \(\'nudi_handfish_auv_v1_train\',\'nudi_handfish_auv_v2_train\',\'nudi_handfish_rov_v3_train\',\'trench_handfish_auv_v1_train\',\) DATASETS.TEST \(\'squidle_handfish_15800_test\',\) LOGGING.GROUP_TAGS HI2SQ,H1 "
sbatch --partition=accel tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/handfishinf/base_strongaug_ema_sq/\' DATASETS.TRAIN \(\'nudi_handfish_auv_v1_train\',\'nudi_handfish_auv_v2_train\',\'nudi_handfish_rov_v3_train\',\'trench_handfish_auv_v1_train\',\) DATASETS.TEST \(\'squidle_handfish_15800_test\',\) LOGGING.GROUP_TAGS HI2SQ,H1 "
sbatch --partition=accel tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_weakaug.yaml "OUTPUT_DIR \'outputs/handfishinf/base_weakaug_sq_nowater/\' DATASETS.TRAIN \(\'nudi_handfish_auv_v1_train\',\'nudi_handfish_auv_v2_train\',\'nudi_handfish_rov_v3_train\',\'trench_handfish_auv_v1_train\',\'nudi_handfish_auv_v1_train_nowater\',\'nudi_handfish_auv_v2_train_nowater\',\'nudi_handfish_rov_v3_train_nowater\',\'trench_handfish_auv_v1_train_nowater\',\) DATASETS.TEST \(\'squidle_handfish_15800_test\',\) LOGGING.GROUP_TAGS HI2SQ,H1 "
sbatch --partition=accel tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/handfishinf/base_strongaug_ema_sq_nowater/\' DATASETS.TRAIN \(\'nudi_handfish_auv_v1_train\',\'nudi_handfish_auv_v2_train\',\'nudi_handfish_rov_v3_train\',\'trench_handfish_auv_v1_train\',\'nudi_handfish_auv_v1_train_nowater\',\'nudi_handfish_auv_v2_train_nowater\',\'nudi_handfish_rov_v3_train_nowater\',\'trench_handfish_auv_v1_train_nowater\',\) DATASETS.TEST \(\'squidle_handfish_15800_test\',\) LOGGING.GROUP_TAGS HI2SQ,H1 "


# Oracle - Weak / Strong
sbatch --partition=accel tools/saga_slurm_train_net.sh OracleT-RCNN-FPN-urchininf_weakaug.yaml "OUTPUT_DIR \'outputs/handfishinf/oracle_weakaug_sq/\' DATASETS.TRAIN \(\'squidle_handfish_15800_train\',\) DATASETS.TEST \(\'squidle_handfish_15800_test\',\) LOGGING.GROUP_TAGS Sq,H1"
sbatch --partition=accel tools/saga_slurm_train_net.sh OracleT-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/handfishinf/oracle_strongaug_ema_sq/\' DATASETS.TRAIN \(\'squidle_handfish_15800_train\',\) DATASETS.TEST \(\'squidle_handfish_15800_test\',\) LOGGING.GROUP_TAGS Sq,H1"


# EVAL
sbatch --partition=accel tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema.yaml --eval-only 

# DAOD methods with unlabelled target data
sbatch --partition=accel tools/saga_slurm_train_net.sh MeanTeacher-urchininf.yaml "OUTPUT_DIR \'outputs/handfishinf/mt_inf_sq/\' DATASETS.TEST \(\'squidle_handfish_15800_test\',\) DATASETS.TRAIN \(\'nudi_handfish_auv_v1_train\',\'nudi_handfish_auv_v2_train\',\'nudi_handfish_rov_v3_train\',\'trench_handfish_auv_v1_train\',\) DATASETS.UNLABELED \(\'squidle_handfish_15800_train\',\) LOGGING.GROUP_TAGS HI2SQ,H1 MODEL.WEIGHTS outputs/handfishinf/base_strongaug_ema_sq/`cat outputs/handfishinf/base_strongaug_ema_sq/last_checkpoint`  "
sbatch --partition=accel tools/saga_slurm_train_net.sh ALDI-urchininf.yaml "OUTPUT_DIR \'outputs/handfishinf/aldi_inf_sq/\' DATASETS.TEST \(\'squidle_handfish_15800_test\',\) DATASETS.TRAIN \(\'nudi_handfish_auv_v1_train\',\'nudi_handfish_auv_v2_train\',\'nudi_handfish_rov_v3_train\',\'trench_handfish_auv_v1_train\',\) DATASETS.UNLABELED \(\'squidle_handfish_15800_train\',\) LOGGING.GROUP_TAGS HI2SQ,H1 MODEL.WEIGHTS outputs/handfishinf/base_strongaug_ema_sq/`cat outputs/handfishinf/base_strongaug_ema_sq/last_checkpoint` "
sbatch --partition=accel tools/saga_slurm_train_net.sh urchininf_priorart/AT-urchininf.yaml "OUTPUT_DIR \'outputs/handfishinf/at_inf_sq/\' DATASETS.TEST \(\'squidle_handfish_15800_test\',\) DATASETS.TRAIN \(\'nudi_handfish_auv_v1_train\',\'nudi_handfish_auv_v2_train\',\'nudi_handfish_rov_v3_train\',\'trench_handfish_auv_v1_train\',\) DATASETS.UNLABELED \(\'squidle_handfish_15800_train\',\) LOGGING.GROUP_TAGS HI2SQ,H1 MODEL.WEIGHTS outputs/handfishinf/base_strongaug_ema_sq/`cat outputs/handfishinf/base_strongaug_ema_sq/last_checkpoint` "
sbatch --partition=accel tools/saga_slurm_train_net.sh urchininf_priorart/SADA_urchininf.yaml "OUTPUT_DIR \'outputs/handfishinf/sada_inf_sq/\' DATASETS.TEST \(\'squidle_handfish_15800_test\',\) DATASETS.TRAIN \(\'nudi_handfish_auv_v1_train\',\'nudi_handfish_auv_v2_train\',\'nudi_handfish_rov_v3_train\',\'trench_handfish_auv_v1_train\',\) DATASETS.UNLABELED \(\'squidle_handfish_15800_train\',\) LOGGING.GROUP_TAGS HI2SQ,H1  MODEL.WEIGHTS outputs/handfishinf/base_strongaug_ema_sq/`cat outputs/handfishinf/base_strongaug_ema_sq/last_checkpoint` "

# Base Train - with labelled target data. No DAOD
sbatch --partition=accel tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/handfishinf/base_inf_sqall/\' DATASETS.TEST \(\'squidle_handfish_15800_test\',\) DATASETS.TRAIN \(\'nudi_handfish_auv_v1_train\',\'nudi_handfish_auv_v2_train\',\'nudi_handfish_rov_v3_train\',\'trench_handfish_auv_v1_train\',\'squidle_handfish_15800_train\',\) LOGGING.GROUP_TAGS HI2SQ,H1,ALL "
sbatch --partition=accel tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema.yaml "OUTPUT_DIR \'outputs/handfishinf/base_inf_sqall/\' DATASETS.TEST \(\'squidle_handfish_15800_test\',\) DATASETS.TRAIN \(\'nudi_handfish_auv_v1_train\',\'nudi_handfish_auv_v2_train\',\'nudi_handfish_rov_v3_train\',\'trench_handfish_auv_v1_train\',\'nudi_handfish_auv_v1_train_nowater\',\'nudi_handfish_auv_v2_train_nowater\',\'nudi_handfish_rov_v3_train_nowater\',\'trench_handfish_auv_v1_train_nowater\',\'squidle_handfish_15800_train\',\) LOGGING.GROUP_TAGS HI2SQ,H1,ALL_NW "


####################### NEEDS WORK ################################################
#### DAOD
### WITH STRONGAUG PRETRAIN
# MeanTeacher Inf With some Squidle target labeled data - Inf+no sq, 200, 400, 600, All sq+Inf
sbatch --partition=accel tools/saga_slurm_train_net.sh MeanTeacher-urchininf_sq.yaml "OUTPUT_DIR \'outputs/handfishinf/mt_inf_sq/\' DATASETS.TEST \(\'squidle_handfish_15800_test\',\) DATASETS.TRAIN \(\'nudi_handfish_auv_v1_train\',\'nudi_handfish_auv_v2_train\',\'nudi_handfish_rov_v3_train\',\'trench_handfish_auv_v1_train\',\)  DATASETS.UNLABELED \(\'squidle_handfish_15800_train\',\) LOGGING.GROUP_TAGS HI2SQ,H1"
sbatch  --partition=accel  tools/saga_slurm_train_net.sh MeanTeacher-urchininf_sq.yaml "OUTPUT_DIR \'outputs/handfishinf/mt_inf_sqall/\' LOGGING.GROUP_TAGS Inf+alltgt2SQ,H1  DATASETS.TRAIN \(\'nudi_handfish_auv_v1_train\',\'squidle_handfish_15800_train\',\) "

# ALDI Inf With some Squidle target labeled data - Inf+no sq, 200, 400, 600, All sq+Inf
sbatch --partition=accel tools/saga_slurm_train_net.sh ALDI-urchininf_sq.yaml "OUTPUT_DIR \'outputs/handfishinf/aldi_inf_sq/\' LOGGING.GROUP_TAGS HI2SQ,H1 "
sbatch  --partition=accel  tools/saga_slurm_train_net.sh ALDI-urchininf_sq.yaml "OUTPUT_DIR \'outputs/handfishinf/aldi_inf_sqall/\' LOGGING.GROUP_TAGS Inf+alltgt2SQ,H1 DATASETS.TRAIN \(\'nudi_handfish_auv_v1_train\',\'squidle_handfish_15800_train\',\) "


# ALDI - with best model from Base Training - Assumes model selection
sbatch --partition=accel tools/saga_slurm_train_net.sh ALDI-urchininf.yaml "DATASETS.TRAIN \(\'nudi_handfish_auv_v1_train\',\'nudi_handfish_auv_v2_train\',\'nudi_handfish_rov_v3_train\',\'trench_handfish_auv_v1_train\',\)  DATASETS.TEST \(\'squidle_handfish_15800_test\',\) DATASETS.UNLABELED \(\'squidle_handfish_15800_train\',\) LOGGING.GROUP_TAGS HI2SQ,H1,BestP MODEL.WEIGHTS \'outputs/handfishinf/urchininf_base_strongaug_ema/squidle_urchin_2011_test_model_best.pth\'"

