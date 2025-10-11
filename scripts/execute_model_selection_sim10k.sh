#!/bin/bash

# python tools/run_model_selection.py --config-file configs/urchininf/
# sbatch --partition=accel tools/saga_slurm_model_selection.sh 

script_start="python tools/"
hd="/home/ubuntu"
gd_source="aldi0107"


datasets=("cityscapes/cityscapes_baseline_strongaug_ema")
for model_dataset in "${datasets[@]}"
do
  ${script_start}run_model_selection.py --config-file "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/config.yaml" OUTPUT_DIR "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/" SOLVER.IMS_PER_BATCH 8 LOGGING.GROUP_TAGS MS4
  #${script_start}modelSele_DAS.py --config-file "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/config.yaml" OUTPUT_DIR "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/" SOLVER.IMS_PER_BATCH 8 LOGGING.GROUP_TAGS MS4
  ${script_start}BoS_test.py --config-file "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/config.yaml" OUTPUT_DIR "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/" LOGGING.GROUP_TAGS MS4
done

datasets=("sim10k/sim10k_baseline_strongaug_ema" \
  "sim10k/sim10k_ALDI_final" "cityscapes/cityscapes_aldi_final" \
  "sim10k/sim10k_MT_final" "cityscapes/cityscapes_meanteacher_final")
#datasets=("cityscapes/cityscapes_baseline_strongaug_ema")
for model_dataset in "${datasets[@]}"
do
  ${script_start}run_model_selection.py --config-file "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/config.yaml" OUTPUT_DIR "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/" SOLVER.IMS_PER_BATCH 8 LOGGING.GROUP_TAGS MS4
  ${script_start}modelSele_DAS.py --config-file "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/config.yaml" OUTPUT_DIR "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/" SOLVER.IMS_PER_BATCH 8 LOGGING.GROUP_TAGS MS4
  ${script_start}BoS_test.py --config-file "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/config.yaml" OUTPUT_DIR "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/" LOGGING.GROUP_TAGS MS4
done