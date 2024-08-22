#!/bin/bash

#OVERRIDE='DATASETS.TRAIN ("squidle_urchin_2009_train",) DATASETS.TEST ("squidle_urchin_2011_test",) DATASETS.UNLABELED ("squidle_urchin_2011_test",)'
OVERRIDE='MODEL.ROI_HEADS.NUM_CLASSES 4 DATASETS.TRAIN ("SUODAC2020_train",) DATASETS.TEST ("SUODAC2020_test",) DATASETS.UNLABELED ("SUODAC2020_test",) LOGGING.GROUP_TAGS "SUODAC2020,S4" SOLVER.BASE_LR 0.08 SOLVER.IMS_PER_GPU 4'
#OVERRIDE=""
GROUP_TAGS="" #'LOGGING.GROUP_TAGS "Inf2UDD,S2"'
echo $OVERRIDE $GROUP_TAGS
python tools/train_net.py --config configs/urchininf/Base-RCNN-FPN-urchininf_strongaug_ema.yaml  ${OVERRIDE} ${GROUP_TAGS} #SOLVER.IMS_PER_GPU 4
#python tools/train_net.py --config configs/urchininf/ALDI-urchininf.yaml ${OVERRIDE} ${GROUP_TAGS} #SOLVER.IMS_PER_GPU 4
#python tools/train_net.py --config configs/urchininf/Base-RCNN-FPN-urchininf_weakaug.yaml ${OVERRIDE} ${GROUP_TAGS} SOLVER.IMS_PER_GPU 4
#python tools/train_net.py --config configs/urchininf/MeanTeacher-urchininf.yaml ${OVERRIDE} ${GROUP_TAGS}
#python tools/train_net.py --config configs/urchininf/urchininf_priorart/AT-urchininf.yaml ${OVERRIDE} ${GROUP_TAGS} SOLVER.IMS_PER_GPU 4
#python tools/train_net.py --config configs/urchininf/OracleT-RCNN-FPN-urchininf_strongaug_ema.yaml ${OVERRIDE} ${GROUP_TAGS} SOLVER.IMS_PER_GPU 4
#python tools/train_net.py --config configs/urchininf/urchininf_priorart/SADA_urchininf.yaml ${OVERRIDE} ${GROUP_TAGS} SOLVER.IMS_PER_GPU 4 SOLVER.BASE_LR 0.003
#python tools/train_net.py --config configs/urchininf/urchininf_priorart/MIC-urchininf_NoBurnIn.yaml ${OVERRIDE} ${GROUP_TAGS} SOLVER.BASE_LR 0.003
