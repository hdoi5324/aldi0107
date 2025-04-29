#!/bin/bash

# python tools/run_model_selection.py --config-file configs/urchininf/
# sbatch --partition=accel tools/saga_slurm_model_selection.sh 


# Sim10k - cityscapes val
python tools/run_model_selection.py --config-file configs/sim10k/Base-RCNN-FPN-Sim10k_strongaug_ema.yaml SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 250 MODEL_SELECTION.N_PERTURBATIONS 3 DATASETS.TEST \(\'cityscapes_cars_train\',\'cityscapes_cars_val\',\)
python tools/run_model_selection.py --config-file configs/sim10k/ALDI-Sim10k.yaml SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 250 MODEL_SELECTION.N_PERTURBATIONS 3 DATASETS.TEST \(\'cityscapes_cars_train\',\'cityscapes_cars_val\',\)
python tools/run_model_selection.py --config-file configs/sim10k/MeanTeacher-Sim10k.yaml SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 250 MODEL_SELECTION.N_PERTURBATIONS 3 DATASETS.TEST \(\'cityscapes_cars_train\',\'cityscapes_cars_val\',\)
python tools/run_model_selection.py --config-file configs/sim10k/MeanTeacher-Sim10k_bestpre.yaml SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 250 MODEL_SELECTION.N_PERTURBATIONS 3 DATASETS.TEST \(\'cityscapes_cars_train\',\'cityscapes_cars_val\',\)
python tools/run_model_selection.py --config-file configs/sim10k/ALDI-Sim10k_bestpre.yaml SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 250 MODEL_SELECTION.N_PERTURBATIONS 3 DATASETS.TEST \(\'cityscapes_cars_train\',\'cityscapes_cars_val\',\)
python tools/run_model_selection.py --config-file configs/sim10k/Base-RCNN-FPN-Sim10k.yaml SEED 1234575 LOGGING.TAGS MS LOGGING.GROUP_TAGS MS3 MODEL_SELECTION.SAMPLE 250 MODEL_SELECTION.N_PERTURBATIONS 3 DATASETS.TEST \(\'cityscapes_cars_train\',\'cityscapes_cars_val\',\)
