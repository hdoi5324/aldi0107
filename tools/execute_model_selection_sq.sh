#!/bin/bash

# python tools/run_model_selection.py --config-file configs/urchininf/
# sbatch --partition=accel tools/saga_slurm_model_selection.sh 

# Oracle - Weak / Strong

# Base Strong urchin - test UDD / Squidle
python tools/run_model_selection.py --config-file configs/urchininf/Base-RCNN-FPN-urchininf_strongaug_ema_sq.yaml SEED 1234575 OUTPUT_DIR \'outputs/urchininf/base_strongaug_ema_inf_sq/\' SOLVER.IMS_PER_BATCH 8 LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 5000 MODEL_SELECTION.N_PERTURBATIONS 3 DATASETS.TRAIN \(\) DATASETS.TEST \(\'UDD_train\',\'UDD_test\',\'squidle_urchin_2011_test\',\)
#python tools/run_model_selection.py --config-file configs/sim10k/Base-RCNN-FPN-urchininf_weakaug.yaml SEED 1234575 OUTPUT_DIR \'outputs/urchininf/base_weakaug_udd/\' SOLVER.IMS_PER_BATCH 8 LOGGING.TAGS MS3 LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 200 MODEL_SELECTION.N_PERTURBATIONS 3 DATASETS.TEST \(\'UDD_train\',\'UDD_test\',\'squidle_urchin_2011_test\',\)

#Squidle
#python tools/run_model_selection.py --config-file configs/urchininf/ALDI-urchininf_sq.yaml SEED 1234575 OUTPUT_DIR \'outputs/urchininf/aldi_inf_sq/\' SOLVER.IMS_PER_BATCH 8 LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 200 MODEL_SELECTION.N_PERTURBATIONS 3 DATASETS.TEST \(\'squidle_urchin_2011_test\',\'squidle_urchin_2009_train\',\'squidle_east_tas_urchins_train\',\)
#python tools/run_model_selection.py --config-file configs/urchininf/MeanTeacher-urchininf_sq.yaml SEED 1234575 OUTPUT_DIR \'outputs/urchininf/mt_inf_sq/\' SOLVER.IMS_PER_BATCH 8 LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 200 MODEL_SELECTION.N_PERTURBATIONS 3 DATASETS.TEST \(\'squidle_urchin_2011_test\',\'squidle_urchin_2009_train\',\'squidle_east_tas_urchins_train\',\)

#UDD
#python tools/run_model_selection.py --config-file configs/urchininf/ALDI-urchininf.yaml SEED 1234575 OUTPUT_DIR \'outputs/urchininf/aldi_inf_udd/\' SOLVER.IMS_PER_BATCH 8 LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 200 MODEL_SELECTION.N_PERTURBATIONS 3 DATASETS.TEST \(\'UDD_train\',\'UDD_test\',\)
#python tools/run_model_selection.py --config-file configs/urchininf/MeanTeacher-urchininf.yaml SEED 1234575 OUTPUT_DIR \'outputs/urchininf/mt_inf_udd/\' SOLVER.IMS_PER_BATCH 8 LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 200 MODEL_SELECTION.N_PERTURBATIONS 3 DATASETS.TEST \(\'UDD_train\',\'UDD_test\',\)

# Baseline Weak
#python tools/run_model_selection.py --config-file configs/sim10k/Base-RCNN-FPN-urchininf_weakaug.yaml SEED 1234575 OUTPUT_DIR \'outputs/urchininf/base_weakaug_udd/\' SOLVER.IMS_PER_BATCH 8 LOGGING.TAGS MS3 LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 200 MODEL_SELECTION.N_PERTURBATIONS 3 DATASETS.TEST \(\'UDD_train\',\'UDD_test\',\'squidle_urchin_2011_test\',\)
