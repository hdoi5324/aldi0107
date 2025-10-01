#!/bin/bash

ims_per_gpu=16
#python tools/train_net.py --config-file configs/sim10k/Base-RCNN-FPN-Sim10k_strongaug_ema.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS3

# ALDI S2C
#python tools/train_net.py --config-file configs/sim10k/ALDI-Sim10k_final.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS3
#python tools/train_net.py --config-file configs/sim10k/ALDI-Sim10k_max.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS3
python tools/train_net.py --config-file configs/sim10k/ALDI-Sim10k_ums.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS3

ims_per_gpu=16
#python tools/train_net.py --config-file configs/cityscapes/Base-RCNN-FPN-Cityscapes_strongaug_ema.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS C2FC,UMS3

# ALDI C2FC
#python tools/train_net.py --config-file configs/cityscapes/ALDI-Cityscapes_final.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS C2FC,UMS3
#python tools/train_net.py --config-file configs/cityscapes/ALDI-Cityscapes_max.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS C2FC,UMS3
python tools/train_net.py --config-file configs/cityscapes/ALDI-Cityscapes_ums.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS C2FC,UMS3

# MT S2C
ims_per_gpu=24
#python tools/train_net.py --config-file configs/sim10k/MeanTeacher-Sim10k_final.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS3
python tools/train_net.py --config-file configs/sim10k/MeanTeacher-Sim10k_max.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS3
#python tools/train_net.py --config-file configs/sim10k/MeanTeacher-Sim10k_ums.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS S2C,UMS3

# MT C2FC
ims_per_gpu=24
#python tools/train_net.py --config-file configs/cityscapes/MeanTeacher-Cityscapes_final.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS C2FC,UMS3
#python tools/train_net.py --config-file configs/cityscapes/MeanTeacher-Cityscapes_max.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS C2FC,UMS3
#python tools/train_net.py --config-file configs/cityscapes/MeanTeacher-Cityscapes_ums.yaml SOLVER.IMS_PER_GPU ${ims_per_gpu} SEED 1234575 LOGGING.GROUP_TAGS C2FC,UMS3


