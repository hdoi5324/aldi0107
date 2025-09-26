#from adapteacher.engine.trainer import ATeacherTrainer
from typing import Tuple
import pprint

import numpy as np
from detectron2.layers import batched_nms
from mistune.toc import render_toc_ul
from scipy.optimize import linear_sum_assignment
import copy
import torch
import torch.nn.functional as F
from fvcore.nn.giou_loss import giou_loss
from detectron2.evaluation import inference_on_dataset
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import pairwise_iou, Boxes
from detectron2.modeling.matcher import Matcher

def perturb_model_parameters(module):
    if not hasattr(module, 'original_params'):
        module.original_params = None
    if module.original_params is None: # Perturb model once
        ignoreNames = "D_img"

        stds = []
        print("saving original parameters...")
        module.original_params = {}
        for name, param in module.named_parameters():
            if ignoreNames in name:
                continue
            module.original_params[name] = param.clone()
            std = param.std().item()
            stds.append(std) if not np.isnan(std) else ...
        # print(np.mean(stds))
        step = 1 # * np.exp(np.mean(std))
        print("step is setted to {}".format(step))

        n_params = sum([
            p.numel() for n, p in module.named_parameters() if not ignoreNames in n
        ])
        random_vector = torch.rand(n_params)
        direction = (random_vector / torch.norm(random_vector)).cuda() * step

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


def get_propossal_idx(x, net):
    # Random batch index .
    _, features, proposals, _ = x
    inputs_a = pooled_features(features, proposals, net)

    # Obtain model predictions and hard pseudo labels .
    boxes, scores, image_shape = pred_roi_pooled_features(inputs_a, proposals,
                                                          net)  # network output for multiple images
    # todo: Maybe update for batchsize > 1.  Currently assumes single image.
    keep_idx = fast_rcnn_inference_single_image(boxes[0], scores[0], image_shape[0]) 
    return proposals, keep_idx
    
    
def FIS(model, dataloader, evaluator, max_repeat=1, bos=False):
    #evaluator = ATeacherTrainer.build_evaluator(cfg, cfg.DATASETS.TEST[0])
    #dataloader = dataloaders[1]
    torch.set_grad_enabled(False)

    #res, preds_gallery = ATeacherTrainer.test(cfg, model, evaluators=[evaluator],
    #                                          data_loader=dataloader, perturb=False)
    #data_loader = ALDITrainer.build_test_loader(cfg, dataset)
    
    preds_gallery = []
    #proposals_gallery = []
    forward_hook_handle = model.register_forward_hook(lambda module, input, output: preds_gallery.append(output))
    # todo: add hook to get proposals for the resulting boxes.
    #proposal_hook_handle = model.roi_heads.register_forward_hook(
    #        lambda module, input, output: proposals_gallery.append(get_propossal_idx(input, module)))
    _ = inference_on_dataset(model, dataloader, evaluator=evaluator)
    forward_hook_handle.remove()
    #proposal_hook_handle.remove()
    
    reg_list = [x[0]["instances"].get("pred_boxes").tensor for x in preds_gallery]
    logits = [x[0]["instances"].get("scores") for x in preds_gallery]    # Notion: Softmax before returning.

    results_bbox_per_img = []
    results_logits_per_img = []
    for reg in reg_list:
        bboxes_per_img = [reg[i].cpu().numpy() for i in range(len(reg))]
        results_bbox_per_img.append(bboxes_per_img)
    for logit in logits:
        logits_per_img = [logit[i].cpu().numpy() for i in range(len(logit))]
        results_logits_per_img.append(logits_per_img)

    cost_final_perModel, iou_cost_perModel, giou_cost_perModel, match_quality_perModel = {}, {}, {}, {}
    lambdas = [1] * max_repeat # key used to collect results
    results_dict = {}
    for key in lambdas:
        for cost in ['fis', 'bos_iou_cost', 'giou_cost', 'match_quality', 'same_box_kl_cost']:
            results_dict[f"{cost}_{key}"] = []
        
    for repeat_time in range(max_repeat):
        _lambda = lambdas[repeat_time]
        print("\nrepeat time: {}".format(repeat_time))
        model_copied = copy.deepcopy(model)
        model_copied = perturb_model_parameters(model_copied)
        #_, preds_gallery_perturb = ATeacherTrainer.test(cfg, model_copied, evaluators=[evaluator], 
        #                                                data_loader=dataloader, perturb=True)
        preds_gallery_perturb = []
        image_level_features=[]
        
        # FYI: Dropout code from BoS as an alternative to perturb of model parameters
        #model.module.backbone.dropout_uncertainty = args.dropout_uncertainty
        #model.module.backbone.drop_layers = args.drop_layers
        #model.module.backbone.drop_nn = nn.Dropout(p=args.dropout_uncertainty)
        
        forward_hook_handle = model_copied.register_forward_hook(lambda module, input, output: preds_gallery_perturb.append(output))
        backbone_hook_handle = model_copied.backbone.register_forward_hook(
            lambda module, input, output: image_level_features.append(output))
        _ = inference_on_dataset(model_copied, dataloader, evaluator=evaluator)
        forward_hook_handle.remove()
        backbone_hook_handle.remove()
        
        reg_list_perturbe = [x[0]["instances"].get("pred_boxes").tensor for x in preds_gallery_perturb]
        logits_perturbe = [x[0]["instances"].get("scores") for x in preds_gallery_perturb]

        results_bbox_per_img_perturbe = []
        results_logits_per_img_perturbe = []
        for reg in reg_list_perturbe:
            bboxes_per_img = [reg[i].cpu().numpy() for i in range(len(reg))]
            results_bbox_per_img_perturbe.append(bboxes_per_img)
        for logit in logits_perturbe:
            logits_per_img = [logit[i].cpu().numpy() for i in range(len(logit))]
            results_logits_per_img_perturbe.append(logits_per_img)

        iou_loss = IoUCost()
        num_flag = 0
        least_iou_cost_final = 0
        least_cost_final = 0
        giou_cost = 0
        match_quality = 0
        logit_cost = 0

        for img_idx in range(len(results_logits_per_img)):


            ### From DAS. See https://github.com/HenryYu23/DAS
            _bboxes = results_bbox_per_img[img_idx]
            _bboxes_perturbe = results_bbox_per_img_perturbe[img_idx]
            _logits = results_logits_per_img[img_idx]
            _logits_perturbe = results_logits_per_img_perturbe[img_idx]

            _bboxes = torch.Tensor(_bboxes)
            _bboxes_perturbe = torch.Tensor(_bboxes_perturbe)
            _logits = torch.Tensor([l.reshape((1,)) for l in _logits])
            _logits_perturbe = torch.Tensor([l.reshape((1,)) for l in _logits_perturbe])

            if len(_bboxes.shape) < 2 or len(_bboxes_perturbe.shape) < 2:
                continue

            sampleCnt, _ = _bboxes.shape
            sampleCntPerturb, _ = _bboxes_perturbe.shape
            max_match = min(sampleCnt, sampleCntPerturb)
            num_flag += 1

            iou_cost = iou_loss(_bboxes_perturbe, _bboxes)
            kl_div = computeKLDivergenceMatrix(_logits_perturbe, _logits)

            costMatrix = iou_cost + kl_div * _lambda
            matched_rowIdx, matched_colIdx = linear_sum_assignment(costMatrix)
            least_cost_final += costMatrix[matched_rowIdx, matched_colIdx].sum().numpy().tolist() / max_match 
            ### End from DAS            
            
            # Match boxes using Faster-RCNN matching quality (from detectron2.modelling.proposal_generator.rpn)
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(Boxes(_bboxes_perturbe), Boxes(_bboxes) ) # gt rows, perturb cols
            match_row, match_col = linear_sum_assignment(match_quality_matrix, maximize=True)
            match_quality += match_quality_matrix[match_row, match_col].sum().numpy().tolist() / max_match

            # From BoS - which doesn't use perturb but dropout
            iou_cost_final = iou_cost
            #iou_matched_row_inds, iou_matched_col_inds = linear_sum_assignment(iou_cost_final)
            least_iou_cost_final += iou_cost_final[match_row, match_col].sum().numpy().tolist() / max_match
            
            # Mine
            giou_cost += giou_loss(_bboxes_perturbe[match_row,:], _bboxes[match_col, :], reduction="mean").item()            
            
            props = [p['instances'] for p in preds_gallery[img_idx]]
            for p in props:
                p.proposal_boxes = p.pred_boxes
            _, orig_box_perturb_logit, _ = scores_for_proposals(image_level_features[img_idx], 
                                                          props,
                                                          model_copied.roi_heads)
            orig_box_perturb_logit = orig_box_perturb_logit[0].cpu()
            logit_cost += computeKLDivergence(orig_box_perturb_logit, torch.hstack((_logits, 1.0 - _logits))).item()
            
            # Mine least_cost_final += costMatrix[match_row, match_col].sum().numpy().tolist() / max_match
            
        # Average over number of images
        least_iou_cost_final /= num_flag
        least_cost_final /= num_flag
        logit_cost /= num_flag
        giou_cost /= num_flag
        match_quality /= num_flag

        results_dict[f"fis_{_lambda}"].append(least_cost_final) # iou cost + kl divergence with perturbed model parameters
        results_dict[f"match_quality_{_lambda}"].append(match_quality) # matching cost used to select matching boxes.
        results_dict[f"bos_iou_cost_{_lambda}"].append(least_iou_cost_final) # iou cost with my box selections
        results_dict[f"giou_cost_{_lambda}"].append(giou_cost) # GIOU cost used in Faster RCNN
        results_dict[f"same_box_kl_cost_{_lambda}"].append(logit_cost) # kl for logits from same boxes, different models
        
        print(f"FIS scores: {pprint.pformat(results_dict)}")
        del model_copied

    return results_dict

def PDR(model, dataloader_target, dataloader_source, semisupnet_dis_type=["p2","p3","p4","p5"]):
    # semisupnet_dis_type = cfg.SEMISUPNET.DIS_TYPE
    USE_BACKBONE_FEATURE = True
    # assert len(dataloaders) == 2
    torch.set_grad_enabled(False)
    model.eval()

    # dataloader_source, dataloader_target = dataloaders
    proto_target, _ = getPrototypes(model, dataloader_target,
                                                         USE_BACKBONE_FEATURE, [semisupnet_dis_type])
    proto_source, _ = getPrototypes(model, dataloader_source,
                                                         USE_BACKBONE_FEATURE, [semisupnet_dis_type])
    
    dist = calCateProtoDistance(proto_source, proto_target)
    
    return {"pdr": dist}

def calCateProtoDistance(source: torch.Tensor, target: torch.Tensor):
    cateNum, featureDim = source.shape

    l2_dist_crossDomain = torch.cdist(source, target)
    l2_dist_sourceInDomain = torch.cdist(source, source)
    l2_dist_targetInDomain = torch.cdist(target, target)

    dist_crossDomain_sameCate = l2_dist_crossDomain.diag().sum() / cateNum
    dist_crossDomain_diffCate = (l2_dist_crossDomain.sum() - l2_dist_crossDomain.diag().sum()) / ((cateNum-1) * cateNum)
    dist_sourceInDomain_diffCate = (l2_dist_sourceInDomain.sum() - l2_dist_sourceInDomain.diag().sum()) / ((cateNum-1) * cateNum)
    dist_targetInDomain_diffCate = (l2_dist_targetInDomain.sum() - l2_dist_targetInDomain.diag().sum()) / ((cateNum-1) * cateNum)

    # print(dist_crossDomain_diffCate, dist_sourceInDomain_diffCate, dist_targetInDomain_diffCate, dist_crossDomain_sameCate)
    div = ((dist_crossDomain_diffCate * dist_sourceInDomain_diffCate * dist_targetInDomain_diffCate / dist_crossDomain_sameCate)).cpu().item()

    return div

def getPrototypes(model, dataloader, useBackboneFeature=True, featureNames=["p6"]):
    prototypes = []
    weights = []
    backboneFeatureList = []

    for idx, inputs in enumerate(dataloader):
        print("\robtaining prototypes: {}".format(idx), end="") if idx % 10 == 0 else ...
        assert not model.training

        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features, None)

        #features = [features[f] for f in featureNames]
        features = [features[f] for f in model.roi_heads.box_in_features]
        backboneFeatureList.append(torch.mean(features[0], dim=(2,3)))

        box_features_backbone = model.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])  # [1000, 512, 7, 7]
        box_features_mlp = model.roi_heads.box_head(box_features_backbone)  # [1000, 1024]

        predictions = model.roi_heads.box_predictor(box_features_mlp)
        category_predictions = F.softmax(predictions[0], dim=1)

        if useBackboneFeature:
            instance_features = F.adaptive_avg_pool2d(box_features_backbone, (1, 1)).flatten(start_dim=1)
        else:
            instance_features = box_features_mlp
        
        instance_features_expand = instance_features.unsqueeze(1)       # [1000, 1, featureDim]
        category_predictions_expand = category_predictions.unsqueeze(2) # [1000, classNum, 1]

        prototypes_per_image = instance_features_expand * category_predictions_expand   # [1000, classNum, featureDim]
        prototypes_sum_per_image = prototypes_per_image.sum(dim=0)      # [classNum, featureDim]
        weights_sum_per_image = category_predictions.sum(dim=0)         # [classNum]

        prototypes.append(prototypes_sum_per_image)
        weights.append(weights_sum_per_image)
    
    backboneFeatureCat = torch.cat(backboneFeatureList, dim=0)

    total_prototype = torch.stack(prototypes).sum(dim=0)
    total_weight = torch.stack(weights).sum(dim=0)

    normalized_prototype = total_prototype / total_weight.unsqueeze(1)
    return normalized_prototype, backboneFeatureCat

def computeKLDivergenceMatrix(logits1: torch.Tensor, logits2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    log_probs1 = logits1.log().unsqueeze(1)
    log_probs2 = logits2.log().unsqueeze(0)
    
    kl_matrix = torch.sum(logits1.unsqueeze(1) * (log_probs1 - log_probs2), dim=-1)

    return kl_matrix

def computeKLDivergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Apply softmax to get probability distributions
    p = F.softmax(p, dim=1)
    q = F.softmax(q, dim=1)

    # Calculate KL divergence
    return F.kl_div(p.log(), q, reduction='batchmean')

def _fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)

def _bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):

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

class IoUCost():
    def __init__(self, iou_mode='iou', weight=1.): 
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bboxes, gt_bboxes):
        overlaps = _bbox_overlaps(bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)

        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight


def pooled_features(features, proposals, net):
    "From ROIHeads._forward_box"
    features = [features[f] for f in net.box_in_features]
    box_features = net.box_pooler(features, [x.proposal_boxes for x in proposals])
    return box_features


def pred_roi_pooled_features(box_features, proposals, net):
    box_features = net.box_head(box_features)
    predictions = net.box_predictor(box_features)
    boxes = net.box_predictor.predict_boxes(predictions, proposals)
    scores = net.box_predictor.predict_probs(predictions, proposals)
    image_shapes = [x.image_size for x in proposals]
    return boxes, scores, image_shapes  # from box_predictor.inference method


def scores_for_proposals(features, proposals, net):
    out = pooled_features(features, proposals, net)
    out = pred_roi_pooled_features(out, proposals, net)
    return out
    

def fast_rcnn_inference_single_image(
        boxes,
        scores,
        image_shape: Tuple[int, int],
        score_thresh=0.05,
        nms_thresh=0.5,
        topk_per_image=100,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    indices = torch.tensor([a for a in range(len(scores))]).to(scores.device).unsqueeze(1)
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        indices = indices[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    indices = indices[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    # boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    # result = Instances(image_shape)
    # result.pred_boxes = Boxes(boxes)
    # result.scores = scores
    # result.pred_classes = filter_inds[:, 1]
    # return result, filter_inds[:, 0]
    return indices[keep]
