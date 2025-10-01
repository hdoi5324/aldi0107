#!/bin/bash

ims_per_gpu=16
python tools/train_net.py --config-file configs/urchininf/Base-RCNN-FPN-urchininf_strongaug_ema_sq.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS INF2S,UMS3
python tools/train_net.py --config-file configs/urchininf/Base-RCNN-FPN-urchininf_strongaug_ema.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS INF2U,UMS3

# ALDI S2C
python tools/train_net.py --config-file configs/urchininf/ALDI-urchininf_sq_ums.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS INF2S,UMS3
python tools/train_net.py --config-file configs/urchininf/ALDI-urchininf_udd_ums.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS INF2U,UMS3

# MT S2C
ims_per_gpu=24
python tools/train_net.py --config-file configs/urchininf/MeanTeacher-urchininf_sq_ums.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS INF2S,UMS3
python tools/train_net.py --config-file configs/urchininf/MeanTeacher-urchininf_udd_ums.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS INF2U,UMS3


