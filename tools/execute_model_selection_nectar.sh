#!/bin/bash

# python tools/run_model_selection.py --config-file configs/urchininf/
# sbatch --partition=accel tools/saga_slurm_model_selection.sh 

# Oracle - Weak / Strong

# Base Strong urchin - test UDD / Squidle
python tools/run_model_selection.py --config-file configs/urchininf/Base-RCNN-FPN-urchininf_strongaug_ema_sq.yaml  SEED 1234575 OUTPUT_DIR \'outputs/urchininf/base_strongaug_ema_inf_sq/\' SOLVER.IMS_PER_BATCH 1 SOLVER.IMS_PER_GPU 1 LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 200 MODEL_SELECTION.N_PERTURBATIONS 1 DATASETS.TEST \(\'UDD_test\',\'squidle_urchin_2011_test\',\)
#Squidle
#python tools/run_model_selection.py --config-file configs/sim10k/Base-RCNN-FPN-urchininf_weakaug_sq.yaml "SEED 1234575 OUTPUT_DIR \'outputs/urchininf/base_weakaug_inf_sq/\' LOGGING.TAGS MS3 LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 200"
#UDD
#python tools/run_model_selection.py --config-file configs/sim10k/Base-RCNN-FPN-urchininf_weakaug.yaml "SEED 1234575 OUTPUT_DIR \'outputs/urchininf/base_weakaug_udd/\' DATASETS.TEST \(\'UDD_test\',\)  LOGGING.TAGS MS3 LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 200"

# Sim10k - cityscapes val
#python tools/run_model_selection.py --config-file configs/sim10k/Base-RCNN-FPN-Sim10k.yaml "SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 500 MODEL_SELECTION.N_PERTURBATIONS 1"
#python tools/run_model_selection.py --config-file configs/sim10k/Base-RCNN-FPN-Sim10k_strongaug_ema.yaml "SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 500 MODEL_SELECTION.N_PERTURBATIONS 1"
#python tools/run_model_selection.py --config-file configs/sim10k/ALDI-Sim10k.yaml "SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 500 MODEL_SELECTION.N_PERTURBATIONS 1"
#python tools/run_model_selection.py --config-file configs/sim10k/MeanTeacher-Sim10k.yaml "SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 500 MODEL_SELECTION.N_PERTURBATIONS 1"
#python tools/run_model_selection.py --config-file configs/sim10k/ALDI-Sim10k_bestpre.yaml "SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 500 MODEL_SELECTION.N_PERTURBATIONS 1"
#python tools/run_model_selection.py --config-file configs/sim10k/MeanTeacher-Sim10k_bestpre.yaml "SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 500 MODEL_SELECTION.N_PERTURBATIONS 1"


#python tools/run_model_selection.py --config-file configs/sim10k/Base-RCNN-FPN-Sim10k_strongaug_ema.yaml "SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 500 MODEL_SELECTION.N_PERTURBATIONS 1 DATASETS.TEST \(\'cityscapes_cars_train\',\'cityscapes_cars_val\',\)"
#python tools/run_model_selection.py --config-file configs/sim10k/Base-RCNN-FPN-Sim10k.yaml "SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 500 MODEL_SELECTION.N_PERTURBATIONS 1 DATASETS.TEST \(\'cityscapes_cars_train\',\'cityscapes_cars_val\',\)"
#python tools/run_model_selection.py --config-file configs/sim10k/ALDI-Sim10k.yaml "SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 500 MODEL_SELECTION.N_PERTURBATIONS 1 DATASETS.TEST \(\'cityscapes_cars_train\',\'cityscapes_cars_val\',\)"
#python tools/run_model_selection.py --config-file configs/sim10k/MeanTeacher-Sim10k.yaml "SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 500 MODEL_SELECTION.N_PERTURBATIONS 1 DATASETS.TEST \(\'cityscapes_cars_train\',\'cityscapes_cars_val\',\)"
#python tools/run_model_selection.py --config-file configs/sim10k/MeanTeacher-Sim10k_bestpre.yaml "SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 500 MODEL_SELECTION.N_PERTURBATIONS 1 DATASETS.TEST \(\'cityscapes_cars_train\',\'cityscapes_cars_val\',\)"
