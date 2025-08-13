#!/bin/bash


method="CoStudent"
method_code="CoSt"

python tools/train_net.py --config-file configs/imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SEED 1234575 LOGGING.GROUP_TAGS SAOD1,RED,ALL,${method_code} SAOD.LABELING_METHOD ${method} OUTPUT_DIR "outputs/saod/red_${method} DATASETS.TRAIN \(\'loose_redcup17647_train_og_only\',\)"
python tools/train_net.py --config-file configs/imosauv/CoStudent-RCNN-FPN-redcup_strongaug_ema.yaml SEED 1234575 LOGGING.GROUP_TAGS SAOD1,RED,1_2,${method_code} SAOD.LABELING_METHOD ${method} OUTPUT_DIR "outputs/saod/red_${method} DATASETS.TRAIN \(\'loose_redcup17648_train_og_only\',\)"

