#!/bin/bash

python tools/probe_featurespace.py --config-file configs/urchininf/ALDI-urchininf_sq.yaml OUTPUT_DIR outputs/urchininf/aldi_inf_sq
python tools/probe_featurespace.py --config-file configs/urchininf/ALDI-urchininf.yaml OUTPUT_DIR outputs/urchininf/aldi_inf_udd
python tools/probe_featurespace.py --config-file configs/urchininf/MeanTeacher-urchininf_sq.yaml OUTPUT_DIR outputs/urchininf/mt_inf_sq
python tools/probe_featurespace.py --config-file configs/urchininf/MeanTeacher-urchininf.yaml OUTPUT_DIR outputs/urchininf/mt_inf_udd
python tools/probe_featurespace.py --config-file configs/urchininf/Base-RCNN-FPN-urchininf_strongaug_ema_sq.yaml OUTPUT_DIR outputs/urchininf/base_strongaug_ema_inf_sq 
python tools/probe_featurespace.py --config-file configs/urchininf/Base-RCNN-FPN-urchininf_strongaug_ema.yaml OUTPUT_DIR outputs/urchininf/base_strongaug_ema_inf_udd

