#!/bin/bash

hd="/home/heather"
gd_source="aldi0107"
cd "${hd}/GitHub/aldi0107"
pwd

script_start="python tools/"
#"sim10k/sim10k_baseline_strongaug_ema" "sim10k/sim10k_ALDI_last"
#"cityscapes/cityscapes_baseline_strongaug_ema" "cityscapes/cityscapes_aldi_last" 
#"imosauv/imosauv_baseline_strongaug_ema" "imosauv/imosauv_ALDI_last"
#"urchininf/urchininf_baseline_strongaug_ema" "urchininf/urchininf_ALDI_last"
#model_dataset="cityscapes/cityscapes_baseline_strongaug_ema"
transformed_source=0
sample_size=500
perturb_method="DAS"

datasets=("sim10k/sim10k_baseline_strongaug_ema" "sim10k/sim10k_ALDI_last" "cityscapes/cityscapes_baseline_strongaug_ema" "cityscapes/cityscapes_aldi_last" "imosauv/imosauv_baseline_strongaug_ema" "imosauv/imosauv_ALDI_last" "urchininf/urchininf_baseline_strongaug_ema" "urchininf/urchininf_ALDI_last")
datasets=("imosauv/imosauv_baseline_strongaug_ema" "urchininf/urchininf_baseline_strongaug_ema" "cityscapes/cityscapes_baseline_strongaug_ema")
datasets=("cityscapes/cityscapes_aldi_last" "imosauv/imosauv_ALDI_last" "urchininf/urchininf_ALDI_last" "sim10k/sim10k_ALDI_last")

datasets=("imosauv/imosauv_baseline_strongaug_ema" "urchininf/urchininf_baseline_strongaug_ema" "cityscapes/cityscapes_baseline_strongaug_ema" "sim10k/sim10k_baseline_strongaug_ema" "cityscapes/cityscapes_aldi_last" "imosauv/imosauv_ALDI_last" "urchininf/urchininf_ALDI_last" "sim10k/sim10k_ALDI_last")

#for model_dataset in "${datasets[@]}"
#do
#  ${script_start}run_model_selection.py --config-file "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/config.yaml" OUTPUT_DIR ${hd}/GitHub/${gd_source}/outputs/${model_dataset}/ MODEL_SELECTION.N_SAMPLE ${sample_size} MODEL_SELECTION.N_TRANSFORMED_SOURCE ${transformed_source}
  #${script_start}modelSele_DAS.py --config-file "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/config.yaml" OUTPUT_DIR "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/" MODEL_SELECTION.N_SAMPLE ${sample_size} MODEL_SELECTION.N_TRANSFORMED_SOURCE ${transformed_source}
  #${script_start}BoS_test.py --config-file "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/config.yaml" OUTPUT_DIR "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/" MODEL_SELECTION.N_SAMPLE ${sample_size} MODEL_SELECTION.N_TRANSFORMED_SOURCE ${transformed_source} MODEL_SELECTION.N_PERTURBATIONS 3 MODEL_SELECTION.PERTURB_TYPE "dropout"
#done

datasets=("cityscapes/cityscapes_baseline_strongaug_ema_0.01" "imosauv/imosauv_baseline_strongaug_ema_0.01" "urchininf/urchininf_baseline_strongaug_ema_0.01" "sim10k/sim10k_baseline_strongaug_ema_0.01")
datasets=("cityscapes/cityscapes_baseline_strongaug_ema_0.04" "imosauv/imosauv_baseline_strongaug_ema_0.04" "sim10k/sim10k_baseline_strongaug_ema_0.04" "urchininf/urchininf_baseline_strongaug_ema_0.04")
for model_dataset in "${datasets[@]}"
do
  ${script_start}run_model_selection.py --config-file "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/config.yaml" OUTPUT_DIR ${hd}/GitHub/${gd_source}/outputs/${model_dataset}/ MODEL_SELECTION.N_SAMPLE ${sample_size} MODEL_SELECTION.N_TRANSFORMED_SOURCE ${transformed_source}
  #${script_start}modelSele_DAS.py --config-file "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/config.yaml" OUTPUT_DIR "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/" MODEL_SELECTION.N_SAMPLE ${sample_size} MODEL_SELECTION.N_TRANSFORMED_SOURCE ${transformed_source}
  #${script_start}BoS_test.py --config-file "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/config.yaml" OUTPUT_DIR "${hd}/GitHub/${gd_source}/outputs/${model_dataset}/" MODEL_SELECTION.N_SAMPLE ${sample_size} MODEL_SELECTION.N_TRANSFORMED_SOURCE ${transformed_source} MODEL_SELECTION.N_PERTURBATIONS 3 MODEL_SELECTION.PERTURB_TYPE "dropout"
done

