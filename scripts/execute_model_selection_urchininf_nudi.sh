#!/bin/bash

# python tools/run_model_selection.py --config-file saga_outputs/outputs/urchininf/
# sbatch --partition=accel tools/saga_slurm_model_selection.sh 
script_start="python tools/"
hd="/home/heather"
gd_source="aldi0107"

datasets=("urchininf/base_strongaug_ema_inf_sq" "urchininf/aldi_inf_sq" "urchininf/mt_inf_sq" "urchininf/oracle_strongaug_ema_sq" "urchininf/base_strongaug_ema_inf_udd" "urchininf/aldi_inf_udd" "urchininf/mt_inf_udd" "urchininf/oracle_strongaug_ema_udd")
for model_dataset in "${datasets[@]}"
do
  ${script_start}BoS_test.py --config-file "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/config.yaml" OUTPUT_DIR "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/" LOGGING.GROUP_TAGS MS4
done
