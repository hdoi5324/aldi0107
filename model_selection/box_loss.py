# Copyright (c) Facebook, Inc. and its affiliates.
import math
from typing import List, Tuple, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch.nn import functional as F

from detectron2.layers import cat, ciou_loss, diou_loss, cross_entropy
from detectron2.structures import Boxes
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
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
    """Calculate the loss based on the difference in class prediction for the gt boxes."""
    images, features, proposals, targets = inputs
    
    # based on _forward_box
    if targets is not None: # Need targets to do this
        #todo: change this to compare to raw scores from gt
        features = [features[f].detach() for f in module.box_in_features]
        
        # todo: change this to iterate through inputs 
        # Get class predictions for gt-boxes
        gt_boxes = [x.gt_boxes for x in targets]
        box_features = module.box_pooler(features, gt_boxes)
        box_features = module.box_head(box_features)
        predictions = module.box_predictor(box_features)
        del box_features
        
        scores, proposal_deltas = predictions
        if scores.shape[0] != len(gt_boxes[0]):
            print(scores.shape[0], len(gt_boxes[0]))
            return None
        else:
            #todo: do same as predict_probs from OutputLayer ie softmax
        
            gt_classes = (
                cat([p.gt_classes for p in targets], dim=0) if len(targets) else torch.empty(0)
            )
            if module.box_predictor.use_sigmoid_ce:
                loss_cls = module.box_predictor.sigmoid_cross_entropy_loss(scores, gt_classes)
                probs = scores.sigmoid()
            else:
                loss_cls = cross_entropy(scores, gt_classes, reduction="mean")    
                probs = F.softmax(scores, dim=-1)    
            return loss_cls.detach().to("cpu").numpy(), probs.to("cpu").numpy()
    else:
        return None

def get_outputs_with_image_id(inputs, outputs):
    """Calculate the loss based on the difference in class prediction for the gt boxes."""
    for i, o in enumerate(outputs):
        o["image_id"] = inputs[0][i]['image_id']
    return outputs
