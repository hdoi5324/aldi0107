#!/bin/bash

SCRIPT='python tools/train_net.py'
POSTSCRIPT='DATASETS.TEST \(\"squidle_urchin_2011_test\",\) DATASETS.UNLABELED \(\"squidle_urchin_2009_train\",\) LOGGING.GROUP_TAGS Inf2SQ,T1 '
#POSTSCRIPT='DATASETS.TEST ("SUODAC2020_test",) DATASETS.UNLABELED ("SUODAC2020_test",)'
echo $SCRIPT $POSTSCRIPT 

bash tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_weakaug.yaml DATASETS.TEST \(\"squidle_urchin_2011_test\",\) DATASETS.UNLABELED \(\"squidle_urchin_2009_train\",\) LOGGING.GROUP_TAGS Inf2SQ,T1 
bash tools/saga_slurm_train_net.sh Base-RCNN-FPN-urchininf_strongaug_ema.yaml  DATASETS.TEST \(\"squidle_urchin_2011_test\",\) DATASETS.UNLABELED \(\"squidle_urchin_2009_train\",\) LOGGING.GROUP_TAGS Inf2SQ,T1 
bash tools/saga_slurm_train_net.sh MeanTeacher-urchininf.yaml DATASETS.TEST \(\"squidle_urchin_2011_test\",\) DATASETS.UNLABELED \(\"squidle_urchin_2009_train\",\) LOGGING.GROUP_TAGS Inf2SQ,T1 
bash tools/saga_slurm_train_net.sh ALDI-urchininf.yaml DATASETS.TEST \(\"squidle_urchin_2011_test\",\) DATASETS.UNLABELED \(\"squidle_urchin_2009_train\",\) LOGGING.GROUP_TAGS Inf2SQ,T1  SOLVER.IMS_PER_GPU 4
bash tools/saga_slurm_train_net.sh urchininf_priorart/AT-urchininf.yaml DATASETS.TEST \(\"squidle_urchin_2011_test\",\) DATASETS.UNLABELED \(\"squidle_urchin_2009_train\",\) LOGGING.GROUP_TAGS Inf2SQ,T1  SOLVER.IMS_PER_GPU 4
bash tools/saga_slurm_train_net.sh OracleT-RCNN-FPN-urchininf_strongaug_ema.yaml DATASETS.TEST \(\"squidle_urchin_2011_test\",\) DATASETS.UNLABELED \(\"squidle_urchin_2009_train\",\) LOGGING.GROUP_TAGS Inf2SQ,T1  SOLVER.IMS_PER_GPU 4
bash tools/saga_slurm_train_net.sh urchininf_priorart/SADA_urchininf.yaml DATASETS.TEST \(\"squidle_urchin_2011_test\",\) DATASETS.UNLABELED \(\"squidle_urchin_2009_train\",\) LOGGING.GROUP_TAGS Inf2SQ,T1  SOLVER.IMS_PER_GPU 4 SOLVER.BASE_LR 0.003
bash tools/saga_slurm_train_net.sh urchininf_priorart/MIC-urchininf_NoBurnIn.yaml DATASETS.TEST \(\"squidle_urchin_2011_test\",\) DATASETS.UNLABELED \(\"squidle_urchin_2009_train\",\) LOGGING.GROUP_TAGS Inf2SQ,T1  SOLVER.BASE_LR 0.003
