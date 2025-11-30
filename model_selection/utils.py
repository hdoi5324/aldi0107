import json
import os
import pprint

import pandas as pd
import torch
from detectron2.data.catalog import DatasetCatalog
from detectron2.evaluation import DatasetEvaluators
from detectron2.config import get_cfg

from aldi.helpers import Detectron2COCOEvaluatorAdapter
from aldi.config import add_aldi_config
from aldi.config_aldi_only import add_aldi_only_config
from aldi.config_fcos import add_fcos_config
#from aldi.detr.helpers import add_deformable_detr_config

def build_evaluator(cfg, dataset_name, output_folder=None, do_eval=True):
    """Just do COCO Evaluation.
    The evaluation process generates a coco file for the predictions generated which
    can be used as psuedo labels.
    """
    
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "model_selection")
    evaluator = DatasetEvaluators([Detectron2COCOEvaluatorAdapter(dataset_name, output_dir=output_folder)])
    # Update evaluator to only check sampled images if it's a sampled dataset
    if "_sample_" in dataset_name:
        img_keys = [d['image_id'] for d in DatasetCatalog.get(dataset_name)]
        evaluator._evaluators[0]._coco_api.imgs = {k: v for k, v in
                                                   evaluator._evaluators[0]._coco_api.imgs.items() if
                                                   k in img_keys}
        
    evaluator._evaluators[0]._do_evaluation = do_eval

    return evaluator

def dropout_mask_along_channel(weights, p):
    if p > 0:
        dropout = torch.nn.Dropout2d(p=p)
        mask = torch.ones((weights.shape)).float().to(weights.device)
        mask = dropout(mask)
    else:
        mask = torch.ones((weights.shape[0], weights.shape[1], 1, 1)).float().to(weights.device)
        mask = mask.expand_as(weights)
    return mask


def dropout_masks(module, weights_filter, p=.1):
    state_dict = module.state_dict()
    last_layer_parameters = [(k, v) for k, v in state_dict.items() if weights_filter in k]
    mask_dict = {}
    for (k, w) in last_layer_parameters:
        mask_dict[k] = dropout_mask_along_channel(w, p)
    return mask_dict


def perturb_by_dropout(module, layer_name='fpn_output', p=.1, mask_dict={}, n=0, layer_nos=[5], **kwargs):
    """
    Apply dropout to weights with names matching {layer_name}{layer_no}. 
    Assumes there is a .weights for 2d dropout and a bias.
    Update module with state_dict with dropout applied.
    """
    state_dict = module.state_dict()
    mask_dict = mask_dict[n] if n in mask_dict else {}
    for l in layer_nos:
        weight_parameters = [(k, v) for k, v in state_dict.items() if f"{layer_name}{l}.weight" in k]
        bias_parameters = [(k, v) for k, v in state_dict.items() if f"{layer_name}{l}.bias" in k]
        for (k, w) in weight_parameters:
            mask = mask_dict[k] if k in mask_dict else dropout_mask_along_channel(w, p) 
            state_dict[k] = w * mask
            for bk, bias_w in bias_parameters:
                state_dict[bk] = bias_w * mask [:, 0, 0, 0].squeeze()
    incompatiable_keys = module.load_state_dict(state_dict, strict=False)
    print(incompatiable_keys)
    return module


def save_outputs(outputs, evaluation_dir, filename='model_selection.json'):
    #todo: make more generic for Bos and DAS
    flat_results = []
    columns = None
    for m in outputs:
        for ds in outputs[m]:
            if columns is None:
                columns = ["model", "dataset"] + list(outputs[m][ds].keys())
            model_dataset = [m, ds]
            data = [v for v in outputs[m][ds].values()]
            flat_results.append(model_dataset + data)
    df = pd.DataFrame(flat_results, columns=columns)
    for l in df.columns:
        if 'loss' in l:
            df[f"{l}_std"] = df[l].apply(lambda x: x[1])
            df[l] = df[l].apply(lambda x: x[0])
    output_file = os.path.join(evaluation_dir, filename)
    df.to_json(output_file)
    return output_file


def save_results_dict(results_dict, output_dir, measure_name=""):
    pprint.pp(results_dict)
    filename = os.path.join(output_dir, f"{measure_name}_dict_output.json")
    with open(filename, 'w') as file:
        json.dump(results_dict, file)
    return filename


def _fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def _bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Based on mmdet/structures/bbox/bbox_overlaps.py via BoS implementation.
    
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

        wh = _fp16_clamp(rb - lt, min=0)
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

        wh = _fp16_clamp(rb - lt, min=0)
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
    enclose_wh = _fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def setup(args):
    """
    Copied from detectron2/tools/train_net.py
    """
    cfg = get_cfg()

    ## Change here
    add_aldi_config(cfg)
    add_fcos_config(cfg)
    
    try:
        import aldi.detr.align
        import aldi.detr.distill
        from aldi.detr.helpers import add_deformable_detr_config
        add_deformable_detr_config(cfg)
    except ImportError:
        print("Failed to load DETR.  Skipping...")

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def perturb_model_parameters(module):
    """
    Based on DAS - https://github.com/HenryYu23/DAS.  
    rcnn.py in DAobjTwoStagePseudoLabGeneralizedRCNN.inference method.
    Applies perturbation to model rather than in the model class."""
    
    if not hasattr(module, 'original_params'):
        module.original_params = None
    if module.original_params is None: # Perturb model once
        ignoreNames = "D_img" # Discriminator parameters
        print("saving original parameters...")
        module.original_params = {}
        for name, param in module.named_parameters():
            if ignoreNames in name:
                print(" ***************************************8 Found a discriminator")
                print("***************************************************************")
            module.original_params[name] = param.clone()

        n_params = sum([
            p.numel() for n, p in module.named_parameters() if not ignoreNames in n
        ])
        random_vector = torch.rand(n_params)
        direction = (random_vector / torch.norm(random_vector)).cuda() 
        offset = 0
        for n, p in module.named_parameters():
            if ignoreNames in n:
                continue
            size = p.numel()
            ip = direction[offset:offset+size].view(p.shape)
            p.data = module.original_params[n] + ip
            offset += size
        print("Finished perturbing.")
    return module
