# Copyright (c) Facebook, Inc. and its affiliates.
import math
from typing import List, Tuple, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch.nn import functional as F

from detectron2.layers import cat, ciou_loss, diou_loss, cross_entropy
from detectron2.structures import Boxes
from detectron2.modeling.box_regression import Box2BoxTransform

from aldi.pseudolabeler import process_pseudo_label

# Value for clamping large dw and dh predictions. The heuristic is that we clamp
# such that dw and dh are no larger than what would transform a 16px box into a
# 1000px box (based on a small anchor, 16px, and a typical image size, 1000px).
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)

def _mean_dense_box_regression_loss(
    anchors: List[Union[Boxes, torch.Tensor]],
    box2box_transform: Box2BoxTransform,
    pred_anchor_deltas: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    fg_mask: torch.Tensor,
    box_reg_loss_type="smooth_l1",
    smooth_l1_beta=0.0,
):
    """
    Compute loss for dense multi-level box regression.
    Loss is accumulated over ``fg_mask``.

    Args:
        anchors: #lvl anchor boxes, each is (HixWixA, 4)
        pred_anchor_deltas: #lvl predictions, each is (N, HixWixA, 4)
        gt_boxes: N ground truth boxes, each has shape (R, 4) (R = sum(Hi * Wi * A))
        fg_mask: the foreground boolean mask of shape (N, R) to compute loss on
        box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou",
            "diou", "ciou".
        smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
            use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
    """
    if isinstance(anchors[0], Boxes):
        anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
    else:
        anchors = cat(anchors)
    if box_reg_loss_type == "smooth_l1":
        gt_anchor_deltas = [box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)
        loss_box_reg = smooth_l1_loss(
            cat(pred_anchor_deltas, dim=1)[fg_mask],
            gt_anchor_deltas[fg_mask],
            beta=smooth_l1_beta,
            reduction="sum",
        )
    elif box_reg_loss_type == "giou":
        pred_boxes = [
            box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
        ]
        loss_box_reg = giou_loss(
            torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction="mean"
        )
    elif box_reg_loss_type == "diou":
        pred_boxes = [
            box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
        ]
        loss_box_reg = diou_loss(
            torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction="mean"
        )
    elif box_reg_loss_type == "ciou":
        pred_boxes = [
            box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
        ]
        loss_box_reg = ciou_loss(
            torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction="mean"
        )
    else:
        raise ValueError(f"Invalid dense box regression loss type '{box_reg_loss_type}'")
    return loss_box_reg


def classifier_loss_on_gt_boxes(module, inputs):
    """Called on StandardROIHeads to calculate classifier loss for no change in box proposals"""
    _, features, proposals, targets = inputs
    
    proposals = module.label_and_sample_proposals(proposals, targets)
    del targets
    
    filtered_props = []
    for p in proposals:
        if p.has('gt_boxes'):
            filtered_props.append(p)
        else:
            p.set('gt_boxes', p.proposal_boxes.clone())
            filtered_props.append(p)
                  
    proposals = filtered_props # filter where there are no boxes

    # based on _forward_box
    features = [features[f].detach() for f in module.box_in_features]
    gt_boxes = [x.gt_boxes for x in proposals]

    box_features = module.box_pooler(features, gt_boxes)
    box_features = module.box_head(box_features)
    predictions = module.box_predictor(box_features)
    del box_features, features

    scores, proposal_deltas = predictions

    #losses = module.box_predictor.losses(predictions, proposals)
    # parse classification outputs
    gt_classes = (
        cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
    )
    if module.box_predictor.use_sigmoid_ce:
        loss_cls = module.box_predictor.sigmoid_cross_entropy_loss(scores, gt_classes)
    else:
        loss_cls = cross_entropy(scores, gt_classes, reduction="mean")
    return loss_cls.detach().to("cpu").numpy()
