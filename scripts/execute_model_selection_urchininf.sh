#!/bin/bash

# python tools/run_model_selection.py --config-file saga_outputs/outputs/urchininf/
# sbatch --partition=accel tools/saga_slurm_model_selection.sh 
script_start="python tools/"
hd="/home/heather"
gd_source="aldi0107/saga_outputs"

datasets=("urchininf/base_strongaug_ema_inf_sq" "urchininf/aldi_inf_sq" "urchininf/mt_inf_sq" "urchininf/oracle_strongaug_ema_sq" "urchininf/base_strongaug_ema_inf_udd" "urchininf/aldi_inf_udd" "urchininf/mt_inf_udd" "urchininf/oracle_strongaug_ema_udd")
datasets=("urchininf/aldi_inf_udd_final" "urchininf/mt_inf_sq_final" "urchininf/mt_inf_udd_final" "urchininf/urchininf_baseline_strongaug_ema_udd" "urchininf/urchininf_baseline_strongaug_ema_sq")
for model_dataset in "${datasets[@]}"
do
  #${script_start}modelSele_DAS.py --config-file "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/config.yaml" OUTPUT_DIR "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/" SOLVER.IMS_PER_BATCH 8 LOGGING.GROUP_TAGS MS4
  #${script_start}BoS_test.py --config-file "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/config.yaml" OUTPUT_DIR "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/" SOLVER.IMS_PER_BATCH 8 LOGGING.GROUP_TAGS MS4
  ${script_start}run_model_selection.py --config-file "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/config.yaml" SEED 1234575 OUTPUT_DIR "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/" SOLVER.IMS_PER_BATCH 8 LOGGING.GROUP_TAGS MS4 
done

