# Everything in this file makes domain adaptation disabled by default.
# Everything must be explicitly enabled in the config files.

from detectron2.config import CfgNode as CN


def add_aldi_only_config(cfg):
    _C = cfg

    # EMA of student weights
    _C.EMA.START_ITER = 0

    _C.DOMAIN_ADAPT.ALIGN.MIXIN_NAME = "AlignMixin"

    # Self-distillation
    _C.DOMAIN_ADAPT.DISTILL.MIXIN_NAME = "DistillMixin"

    # Extra configs for convnext
    # Default is ConvNext-T (Resnet-50 equiv.)
    _C.MODEL.CONVNEXT = CN()
    _C.MODEL.CONVNEXT.DEPTHS= [3, 3, 9, 3]
    _C.MODEL.CONVNEXT.DIMS= [96, 192, 384, 768]
    _C.MODEL.CONVNEXT.DROP_PATH_RATE= 0.2
    _C.MODEL.CONVNEXT.LAYER_SCALE_INIT_VALUE= 1e-6
    _C.MODEL.CONVNEXT.OUT_FEATURES= [0, 1, 2, 3]
    _C.SOLVER.WEIGHT_DECAY_RATE= 0.95
    
    

