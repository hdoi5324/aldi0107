from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.coco import dataloader
from ..common.models.fcos import model
from ..common.train import train

dataloader.train.mapper.use_instance_mask = False
optimizer.lr = 0.01

model.backbone.bottom_up.freeze_at = 2
train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"

from detectron2.config import get_cfg
from detectron2.projects.fcos import add_fcos_config

def setup():
    cfg = get_cfg()
    add_fcos_config(cfg)
    cfg.merge_from_file("path/to/fcos_R_50_FPN_1x.yaml")  # Optional fallback
    cfg.DATASETS.TRAIN = ("coco_2017_train",)
    cfg.DATASETS.TEST = ("coco_2017_val",)
    cfg.OUTPUT_DIR = "./output/fcos"
    cfg.SOLVER.MAX_ITER = 90000
    return cfg
