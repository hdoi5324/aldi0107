# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
import pprint

#import mmcv
import torch
#from mmcv import Config, DictAction
#from mmcv.cnn import fuse_conv_bn
#from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
#                         wrap_fp16_model)

#from mmdet.apis import multi_gpu_test, single_gpu_test
#from mmdet.datasets import (build_dataloader, build_dataset,
#                            replace_ImageToTensor)
#from mmdet.models import build_detector
#from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
#                         replace_cfg_vals, setup_multi_processes,
#                         update_data_root)

#import mmdet_custom
import numpy as np
#import matplotlib.mlab as mlab
#import matplotlib.pyplot as plt
import copy
import glob
from scipy import stats
#import cv2
#import matplotlib.pyplot as plt
#from mmdet.datasets import build_dataset, get_loading_pipeline
#import seaborn as sns

from scipy.optimize import linear_sum_assignment
from detectron2.engine import default_argument_parser, launch
from detectron2.modeling import build_model
from detectron2.data.build import get_detection_dataset_dicts, build_detection_test_loader, DatasetMapper
from detectron2.data.samplers import InferenceSampler
from detectron2.evaluation import inference_on_dataset

# Keep this so that datasets are loaded

from model_selection.utils import perturb_by_dropout, build_evaluator, save_results_dict
from model_selection.model_selection import setup, load_model_weights
from model_selection.fast_rcnn import fast_rcnn_inference_single_image_all_scores


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--tag1',
        default='',
        help='tag1')
    parser.add_argument(
        '--tag2',
        default='',
        help='tag2')
    parser.add_argument(
        '--dropout_uncertainty',
        type=float,
        default=0.01,
        help='tag')
    parser.add_argument(
        '--drop_layers',
        nargs='+', 
        type=int,
        default=[])

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """
    NB Based on from mmdet/structures/bbox/bbox_overlaps.py
    
    Calculate overlap between two set of bboxes.
    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:
        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1
            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,
            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.
            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)
            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB
            When the batch size is B, reduce:
                B x R
            Therefore, CUDA memory runs out frequently.
            Experiments on GeForce RTX 2080Ti (11019 MiB):
            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |
        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1
            Total memory:
                S = 11 x N * 4 Byte
            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte
        So do the 'giou' (large than 'iou').
        Time-wise, FP16 is generally faster than FP32.
        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.
    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.
    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.
    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )
    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def bbox_xyxy_to_cxcywh(bbox):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).
    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.
    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)


def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).
    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.
    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)


class ClassificationCost:
    """ClsSoftmaxCost.
     Args:
         weight (int | float, optional): loss_weight
     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import \
         ... ClassificationCost
         >>> import torch
         >>> self = ClassificationCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3430, -0.3525, -0.3045],
                [-0.3077, -0.2931, -0.3992],
                [-0.3664, -0.3455, -0.2881],
                [-0.3343, -0.2701, -0.3956]])
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
        Returns:
            torch.Tensor: cls_cost value with weight
        """
        # Following the official DETR repo, contrary to the loss that
        # NLL is used, we approximate it in 1 - cls_score[gt_label].
        # The 1 is a constant that doesn't change the matching,
        # so it can be omitted.
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * self.weight


class BBoxL1Cost:
    """BBoxL1Cost.
     Args:
         weight (int | float, optional): loss_weight
         box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN
     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import torch
         >>> self = BBoxL1Cost()
         >>> bbox_pred = torch.rand(1, 4)
         >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(bbox_pred, gt_bboxes, factor)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self, weight=1., box_format='xyxy'):
        self.weight = weight
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                (num_query, 4).
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape (num_gt, 4).
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        if self.box_format == 'xywh':
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        elif self.box_format == 'xyxy':
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


class IoUCost:
    """IoUCost.
     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight
     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode='iou', weight=1.): 
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: iou_cost value with weight
        """
        overlaps = bbox_overlaps(bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)

        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight



def main(args):
    """
    Based on github.com/YangYangGirl/BoS test.py.
    Used the same dataloading and model loading as DAS.
    Use forward hook and override of method for scores_logits.
    """    
    cfg = setup(args)
    model_dirs = sorted(glob.glob(os.path.join(cfg.OUTPUT_DIR, 'model_0*99.pth')))    
    from detectron2.modeling.roi_heads import fast_rcnn 
    fast_rcnn.fast_rcnn_inference_single_image = fast_rcnn_inference_single_image_all_scores    
    
    # Load datasets - #todo: need to update this to load sample datasets
    debug_length=10
    source_dataset = get_detection_dataset_dicts(cfg.DATASETS.TRAIN, filter_empty=False)#[:debug_length]
    dataloader_source = build_detection_test_loader(source_dataset, mapper=DatasetMapper(cfg, False), sampler=InferenceSampler(len(source_dataset)))
    test_dataset = get_detection_dataset_dicts(cfg.DATASETS.TEST[0], filter_empty=False)#[:debug_length]
    dataloader_target = build_detection_test_loader(test_dataset, mapper=DatasetMapper(cfg, False), sampler=InferenceSampler(len(test_dataset)))    
    
    dataloader = dataloader_target
    
    bos_results_dict = {}
    for model_dir in model_dirs: 
        # [MAIN]
        bos_results_dict[model_dir] = {}
        bos_results_dict[model_dir][cfg.DATASETS.TEST[0]] = {}
        # Load model
        model = build_model(cfg)
        model.eval()
        model.training = False
        load_model_weights(model_dir, model) # Loads Teacher (EMA) model if it's present.
        
        evaluator = build_evaluator(cfg, dataset_name=cfg.DATASETS.TEST[0], output_folder=os.path.join(cfg.OUTPUT_DIR, "BoS"))
        torch.set_grad_enabled(False)
        
        # Updated DAS - use forward hook to get preds_gallery
        preds_gallery = []
        forward_hook_handle = model.register_forward_hook(lambda module, input, output: preds_gallery.append(output))
        res = inference_on_dataset(model, dataloader, evaluator=evaluator)
        forward_hook_handle.remove()
    
        results_bbox_per_img = [x[0]["instances"].get("pred_boxes").tensor for x in preds_gallery]
        results_cls_per_img = [x[0]["instances"].get("scores_logits") for x in preds_gallery]    # Notion: Softmax before returning.
        
        # BoS - perturb model through dropout
        model_copied = copy.deepcopy(model)
        # Updated DAS - apply perturbation to model rather than in model code
        model_copied = perturb_by_dropout(model_copied, p=0.1)
        #todo: perturb with dropout
         # Updated DAS - use forward hook to get preds_gallery
        preds_gallery_perterb = []
        forward_hook_handle = model_copied.register_forward_hook(lambda module, input, output: preds_gallery_perterb.append(output))
        _ = inference_on_dataset(model_copied, dataloader, evaluator=evaluator)
        forward_hook_handle.remove()
    
        results_bbox_per_img_perturbe = [x[0]["instances"].get("pred_boxes").tensor for x in preds_gallery_perterb]
        results_cls_per_img_perturbe = [x[0]["instances"].get("scores_logits") for x in preds_gallery_perterb]    # Notion: Softmax before returning.   
        
        reg_loss = BBoxL1Cost()
        iou_loss = IoUCost(iou_mode="giou")  # Paper uses GIoU
        cls_loss = ClassificationCost()
        
        areaRng = [[0 ** 2, 1e5 ** 2]] #[[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        areaRngLbl = ['all', 'small', 'medium', 'large']
        num_flag = [0 for area_idx in areaRng]
    
        #iou_matched = []
        #cls_matched = []
        iou_cost_perturbe = []
        #cls_perturbe = []
        #iou_perturbe_matched = []
        #cls_perturbe_matched = []
        #cost_cls_matched = []
    
        #least_cost = [0 for area_idx in areaRng]
        least_reg_cost_final = [0 for area_idx in areaRng]
        least_iou_cost_final = [0 for area_idx in areaRng]
        #least_cls_cost_final = [0 for area_idx in areaRng]
    
        cls_areaRng = [[] for area_idx in areaRng]
        bboxes_areaRng = []
        entropy_areaRng = [[] for area_idx in areaRng]
            
        for area_idx, area in enumerate(areaRng):
            for img_idx in range(len(results_bbox_per_img)):
                bboxes_raw = results_bbox_per_img[img_idx]
                bboxes_perturbe_raw = results_bbox_per_img_perturbe[img_idx]
                cls_raw = results_cls_per_img[img_idx]
                cls_perturbe_raw = results_cls_per_img_perturbe[img_idx]
    
                bboxes = []
                bboxes_perturbe = []
                cls = []
                cls_perturbe = []
                
                for b_idx, b in enumerate(bboxes_raw):
                    x1, y1, x2, y2 = b
                    w = x2 - x1
                    h = y2 - y1
                    area_b = w * h
                    if area_b > area[0] and area_b < area[1]:
                        bboxes.append(b)
                        cls.append(cls_raw[b_idx])
                        cls_areaRng[area_idx].append(cls_raw[b_idx])
                        entropy_areaRng[area_idx].append(stats.entropy([cls_raw[b_idx].cpu(), 1 - cls_raw[b_idx].cpu()], base=2))
                
                for b_idx, b in enumerate(bboxes_perturbe_raw):
                    x1, y1, x2, y2 = b
                    w = x2 - x1
                    h = y2 - y1
                    area_b = w * h
                    if area_b > area[0] and area_b < area[1]:
                        bboxes_perturbe.append(b)
                        cls_perturbe.append(cls_perturbe_raw[b_idx])
    
                if area_idx == 0:
                    assert len(cls) == len(cls_raw)
                    assert len(cls_perturbe) == len(cls_perturbe_raw)
                    
                if len(bboxes_perturbe) == 0 or len(bboxes) == 0:
                    continue
    
                bboxes = torch.vstack(bboxes)
                #cls = torch.vstack(cls)
                bboxes_perturbe = torch.vstack(bboxes_perturbe)
                #cls_perturbe = torch.vstack(cls_perturbe)
    
                if len(bboxes.shape) < 2 or len(bboxes_perturbe.shape) < 2:
                    continue
                        
                sample, _ = bboxes.shape
                sample_perturbe, _ = bboxes_perturbe.shape
                max_match = min(sample, sample_perturbe)
    
                num_flag[area_idx] += 1
                #img_h, img_w, _ =  dataset.prepare_test_img(img_idx)['img_metas'][0].data['img_shape']
                img_h, img_w = preds_gallery[img_idx][0]["instances"].image_size
                
                factor = bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
                normalize_bboxes =  bboxes / factor
                normalize_bbox_perturbe =  bboxes_perturbe / factor
                normalize_bbox_perturbe = bbox_xyxy_to_cxcywh(normalize_bbox_perturbe)
                reg_cost = reg_loss(normalize_bbox_perturbe, normalize_bboxes)
                iou_cost = iou_loss(bboxes_perturbe, bboxes)
                
                reg_cost_final = reg_cost
                reg_matched_row_inds, reg_matched_col_inds = linear_sum_assignment(reg_cost_final.cpu())
                
                try:
                    least_reg_cost_final[area_idx] += reg_cost_final[reg_matched_row_inds, reg_matched_col_inds].cpu().sum().numpy().tolist() / max_match
                except:
                    import pdb; pdb.set_trace()
    
                #cls_perturbe = torch.transpose(cls_perturbe, 0, 1)
                #cls_cost =  cls_perturbe - cls
                #cls_cost_final = cls_cost
                #cls_matched_row_inds, cls_matched_col_inds = linear_sum_assignment(cls_cost_final)
                #least_cls_cost_final[area_idx] += cls_cost_final[cls_matched_row_inds, cls_matched_col_inds].sum().numpy().tolist() / max_match
    
                #cost_cls_matched.extend(cls_cost_final[cls_matched_row_inds, cls_matched_col_inds].numpy().tolist())
                #cls2cls_matched = cls[0][cls_matched_col_inds]
                #bboxes2cls_matched = bboxes[cls_matched_col_inds]
                #cls_pertube2cls_matched = cls_perturbe[cls_matched_row_inds][0]
    
                #cls_matched.extend(cls2cls_matched.numpy().tolist())
                #cls_perturbe_matched.extend(cls_pertube2cls_matched.numpy().tolist())
    
                iou_cost_final = iou_cost
                iou_matched_row_inds, iou_matched_col_inds = linear_sum_assignment(iou_cost_final.cpu())
                least_iou_cost_final[area_idx] += iou_cost_final[iou_matched_row_inds, iou_matched_col_inds].cpu().sum().numpy().tolist() / max_match
                iou_cost_perturbe.extend(iou_cost_final[iou_matched_row_inds, iou_matched_col_inds].cpu().numpy().tolist())
                    
                #cls_iou_matched = cls[0][iou_matched_col_inds]
                #cls_perturbe_iou_matched = cls_perturbe[iou_matched_row_inds][0]
                bboxes_iou_matched = bboxes[iou_matched_col_inds]
    
                #iou_matched.extend(cls_iou_matched.numpy().tolist())
                #iou_perturbe_matched.extend(cls_perturbe_iou_matched.numpy().tolist())
    
                #try:
                #    cost = 2 * iou_cost + 5 * reg_cost + cls_cost
                #except:
                #    import pdb; pdb.set_trace()
                #matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
                #least_cost[area_idx] += cost[matched_row_inds, matched_col_inds].sum().numpy().tolist() / max_match
    
            #least_cost[area_idx] = least_cost[area_idx] / (num_flag[area_idx])
            least_reg_cost_final[area_idx] = least_reg_cost_final[area_idx] / (num_flag[area_idx])
            least_iou_cost_final[area_idx] = least_iou_cost_final[area_idx] / (num_flag[area_idx])
            #least_cls_cost_final[area_idx] = least_cls_cost_final[area_idx] / (num_flag[area_idx])
        bos_results_dict[model_dir][cfg.DATASETS.TEST[0]]["BoS"] = least_iou_cost_final[0] * -1
        bos_results_dict[model_dir][cfg.DATASETS.TEST[0]]["ground_truth"] = res["bbox"]["AP50"]
        pprint.pp(bos_results_dict)


        # DAS - normalize measures and calculate DAS (sum of FIS and PDR)
    def normalize(array):
        normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
        return normalized_array.tolist()
    
    all_bos = [bos_results_dict[mk][cfg.DATASETS.TEST[0]]["BoS"] for mk in list(bos_results_dict.keys())]
    normalized_bos = normalize(np.array(all_bos))
    for i, mk in enumerate(list(bos_results_dict.keys())):
        bos_results_dict[mk][cfg.DATASETS.TEST[0]][f"BoS_normalized"] = normalized_bos[i]
    
    # Save outputs to file
    bos_results_dict['config_file'] = args.config_file
    
    _ = save_results_dict(bos_results_dict, cfg.OUTPUT_DIR, measure_name="BoS")
        
    #with open(os.path.join(cfg.OUTPUT_DIR, 'DAS_outputs.json'), 'r') as file:
    #    data = json.load(file)
    
    return bos_results_dict


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )