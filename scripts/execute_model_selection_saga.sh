#!/bin/bash

# python tools/model_selection.py --config-file configs/urchininf/
# sbatch --partition=accel tools/saga_slurm_model_selection.sh 


# Base Strong urchin - test UDD / Squidle
#sbatch --partition=accel tools/saga_slurm_model_selection.sh Base-RCNN-FPN-urchininf_strongaug_ema_sq.yaml SEED 1234575 OUTPUT_DIR \'outputs/urchininf/base_strongaug_ema_inf_sq/\' SOLVER.IMS_PER_BATCH 1 SOLVER.IMS_PER_GPU 1 LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 200  DATASETS.TEST \(\'UDD_test\',\'squidle_urchin_2011_test\',\)

#Squidle
#sbatch --partition=accel tools/saga_slurm_model_selection.sh Base-RCNN-FPN-urchininf_weakaug_sq.yaml SEED 1234575 OUTPUT_DIR \'outputs/urchininf/base_weakaug_inf_sq/\' LOGGING.TAGS MS3 LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 200"
#sbatch --partition=accel tools/saga_slurm_model_selection.sh ALDI-urchininf_sq.yaml SEED 1234575 OUTPUT_DIR \'outputs/urchininf/aldi_inf_sq/\' SOLVER.IMS_PER_BATCH 1 SOLVER.IMS_PER_GPU 1 LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 200 
#sbatch --partition=accel tools/saga_slurm_model_selection.sh MeanTeacher-urchininf_sq.yaml SEED 1234575 OUTPUT_DIR \'outputs/urchininf/mt_inf_sq/\' SOLVER.IMS_PER_BATCH 1 SOLVER.IMS_PER_GPU 1 LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 200 

#UDD
#sbatch --partition=accel tools/saga_slurm_model_selection.sh Base-RCNN-FPN-urchininf_weakaug.yaml SEED 1234575 OUTPUT_DIR \'outputs/urchininf/base_weakaug_udd/\' DATASETS.TEST \(\'UDD_test\',\)  LOGGING.TAGS MS3 LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 200"
#sbatch --partition=accel tools/saga_slurm_model_selection.sh Base-RCNN-FPN-urchininf_strongaug_ema.yaml SEED 1234575 OUTPUT_DIR \'outputs/urchininf/base_strongaug_ema_inf_udd/\' SOLVER.IMS_PER_BATCH 1 SOLVER.IMS_PER_GPU 1 LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 200  DATASETS.TEST \(\'UDD_test\',\'squidle_urchin_2011_test\',\)
sbatch --partition=accel tools/saga_slurm_model_selection.sh ALDI-urchininf.yaml SEED 1234575 OUTPUT_DIR \'outputs/urchininf/aldi_inf_udd/\' SOLVER.IMS_PER_BATCH 1 SOLVER.IMS_PER_GPU 1 LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 200 
sbatch --partition=accel tools/saga_slurm_model_selection.sh MeanTeacher-urchininf.yaml SEED 1234575 OUTPUT_DIR \'outputs/urchininf/mt_inf_sq/\' SOLVER.IMS_PER_BATCH 1 SOLVER.IMS_PER_GPU 1 LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 200 

# Sim10k - cityscapes val
#sbatch --partition=accel tools/saga_slurm_model_selection.sh ../sim10k/Base-RCNN-FPN-Sim10k.yaml "SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 500 "
#sbatch --partition=accel tools/saga_slurm_model_selection.sh ../sim10k/Base-RCNN-FPN-Sim10k_strongaug_ema.yaml "SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 500 "
#sbatch --partition=accel tools/saga_slurm_model_selection.sh ../sim10k/ALDI-Sim10k.yaml "SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 500 "
#sbatch --partition=accel tools/saga_slurm_model_selection.sh ../sim10k/MeanTeacher-Sim10k.yaml "SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 500 "
#sbatch --partition=accel tools/saga_slurm_model_selection.sh ../sim10k/ALDI-Sim10k_bestpre.yaml "SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 500 "
#sbatch --partition=accel tools/saga_slurm_model_selection.sh ../sim10k/MeanTeacher-Sim10k_bestpre.yaml "SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 500 "


#sbatch --partition=accel tools/saga_slurm_model_selection.sh ../sim10k/Base-RCNN-FPN-Sim10k_strongaug_ema.yaml SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 500  DATASETS.TEST \(\'cityscapes_cars_train\',\'cityscapes_cars_val\',\)
#sbatch --partition=accel tools/saga_slurm_model_selection.sh ../sim10k/Base-RCNN-FPN-Sim10k.yaml SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 500  DATASETS.TEST \(\'cityscapes_cars_train\',\'cityscapes_cars_val\',\)
#sbatch --partition=accel tools/saga_slurm_model_selection.sh ../sim10k/ALDI-Sim10k.yaml SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 500  DATASETS.TEST \(\'cityscapes_cars_train\',\'cityscapes_cars_val\',\)
#sbatch --partition=accel tools/saga_slurm_model_selection.sh ../sim10k/MeanTeacher-Sim10k.yaml SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 500  DATASETS.TEST \(\'cityscapes_cars_train\',\'cityscapes_cars_val\',\)
#sbatch --partition=accel tools/saga_slurm_model_selection.sh ../sim10k/MeanTeacher-Sim10k_bestpre.yaml SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 UMS.SAMPLE 500  DATASETS.TEST \(\'cityscapes_cars_train\',\'cityscapes_cars_val\',\)
