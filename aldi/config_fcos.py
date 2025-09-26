# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_fcos_config(cfg):
    """
    Add config for FCOS
    """
    _C = cfg

    _C.MODEL.FCOS = CN()
    _C.MODEL.FCOS.NUM_CLASSES = 80
    _C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    _C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
    _C.MODEL.FCOS.PRIOR_PROB = 0.01
    _C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
    _C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
    _C.MODEL.FCOS.NMS_TH = 0.6
    _C.MODEL.FCOS.NMS_TYPE = "iou"
    _C.MODEL.FCOS.SCORE_THRESH_TEST = 0.05
    _C.MODEL.FCOS.TOPK_CANDIDATES_TEST = 1000
    _C.MODEL.FCOS.NMS_THRESH_TEST = 0.6
    
    _C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
    _C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
    _C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
    _C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
    _C.MODEL.FCOS.TOP_LEVELS = 2
    _C.MODEL.FCOS.NORM = "GN"  # Support GN or none
    _C.MODEL.FCOS.USE_SCALE = True

    # Multiply centerness before threshold
    # This will affect the final performance by about 0.05 AP but save some time
    _C.MODEL.FCOS.THRESH_WITH_CTR = False

    # Focal loss parameters
    _C.MODEL.FCOS.LOSS_ALPHA = 0.25
    _C.MODEL.FCOS.LOSS_GAMMA = 2.0
    _C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]

    # the number of convolutions used in the cls and bbox tower
    _C.MODEL.FCOS.NUM_CLS_CONVS = 4
    _C.MODEL.FCOS.NUM_BOX_CONVS = 4
    _C.MODEL.FCOS.NUM_SHARE_CONVS = 0
    _C.MODEL.FCOS.CENTER_SAMPLE = True
    _C.MODEL.FCOS.POS_RADIUS = 1.5
    _C.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
    _C.MODEL.FCOS.YIELD_PROPOSAL = False
    _C.MODEL.FCOS.NUM_PROPOSAL = 700
    _C.MODEL.FCOS.RANDOM_SAMPLE_SIZE = False
    _C.MODEL.FCOS.RANDOM_SAMPLE_SIZE_UPPER_BOUND = 1.0
    _C.MODEL.FCOS.RANDOM_SAMPLE_SIZE_LOWER_BOUND = 0.8
    _C.MODEL.FCOS.RANDOM_PROPOSAL_DROP = False
    _C.MODEL.FCOS.RANDOM_PROPOSAL_DROP_UPPER_BOUND = 1.0
    _C.MODEL.FCOS.RANDOM_PROPOSAL_DROP_LOWER_BOUND = 0.8
    _C.MODEL.FCOS.USE_OBJ_LOSS = False
    _C.MODEL.FCOS.USE_DETR_LOSS = False
    _C.MODEL.FCOS.GIOU_WEIGHT = 4.0
    _C.MODEL.FCOS.PREDICT_WITHOUT_CTR = False
    _C.MODEL.FCOS.EOS_COEF = 0.1
    _C.MODEL.FCOS.ONLY_REWEIGHT_FG = False
    _C.MODEL.FCOS.CLASS_DENORM_TYPE = "all"
    
    _C.MODEL.BACKBONE.NAME = "build_retinanet_resnet_fpn_backbone"
    _C.MODEL.BACKBONE.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
    
    