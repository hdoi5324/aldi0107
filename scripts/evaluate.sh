#!/bin/bash

OVERRIDE='DATASETS.TRAIN ("squidle_urchin_2009_train",) DATASETS.TEST ("squidle_urchin_2011_test",) DATASETS.UNLABELED ("squidle_urchin_2011_test",)'
#OVERRIDE='DATASETS.TEST ("SUODAC2020_test",) '
#OVERRIDE=""
GROUP_TAGS='LOGGING.GROUP_TAGS "Inf2Sq,TEST"'
echo $OVERRIDE $GROUP_TAGS
python tools/visualize_data.py --source dataloader --output-dir outputs/urchininf/base_strongaug_ema_inf_sq1000/squidle_urchin_2011_test --config configs/urchininf/Base-RCNN-FPN-urchininf_strongaug_ema_sq.yaml MODEL.WEIGHTS 'outputs/urchininf/base_strongaug_ema_inf_sq1000/model_final.pth' ${GROUP_TAGS}
python tools/visualize_data.py --source dataloader --output-dir outputs/urchininf/mt_inf_sq/squidle_urchin_2011_test --config configs/urchininf/MeanTeacher-urchininf_sq.yaml MODEL.WEIGHTS 'outputs/urchininf/mt_inf_sq/model_final.pth' ${GROUP_TAGS}
python tools/visualize_data.py --source dataloader --output-dir outputs/urchininf/aldi_inf_sq/squidle_urchin_2011_test --config configs/urchininf/ALDI-urchininf_sq.yaml MODEL.WEIGHTS 'outputs/urchininf/aldi_inf_sq/model_final.pth' ${GROUP_TAGS}
python tools/visualize_data.py --source dataloader --output-dir outputs/urchininf/oracle_strongaug_ema_sq/squidle_urchin_2011_test --config configs/urchininf/OracleT-RCNN-FPN-urchininf_strongaug_ema_sq.yaml MODEL.WEIGHTS 'outputs/urchininf/oracle_strongaug_ema_sq/model_final.pth' ${GROUP_TAGS}
python tools/visualize_data.py --source dataloader --output-dir outputs/urchininf/base_strongaug_ema_inf_udd1000/squidle_urchin_2011_test --config configs/urchininf/Base-RCNN-FPN-urchininf_strongaug_ema.yaml MODEL.WEIGHTS 'outputs/urchininf/base_strongaug_ema_inf_udd1000/model_final.pth' ${GROUP_TAGS}



python tools/visualize_data.py --source dataloader --output-dir outputs/urchininf/mt_inf_udd/udd_test --config configs/urchininf/MeanTeacher-urchininf.yaml MODEL.WEIGHTS 'outputs/urchininf/mt_inf_udd/model_final.pth' ${GROUP_TAGS}

python tools/visualize_data.py --source dataloader --output-dir outputs/redcup/redcup_baseline_strongaug_ema/redcup_test --config ../aldi/configs/imosauv/Base-RCNN-FPN-redcup_strongaug_ema.yaml MODEL.WEIGHTS '../aldi/outputs/redcup/redcup_baseline_strongaug_ema/model_final.pth' LOGGING.GROUP_TAGS "CLW,TEST"
