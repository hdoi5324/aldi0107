import os

#from adapteacher.engine.trainer import ATeacherTrainer

from detectron2.evaluation import inference_on_dataset

from model_selection.fast_rcnn import fast_rcnn_inference_single_image_all_scores
from model_selection.ums import perturb_model_parameters
from model_selection.utils import build_evaluator, _bbox_overlaps
from detectron2.structures import pairwise_iou, Boxes

from scipy.optimize import linear_sum_assignment
import copy
import torch
import torch.nn.functional as F


def FIS(cfg, model, dataloaders, max_repeat=1):
    #todo: update max_repeat code so it works eg captures repeat values then averages them after returned.
    from detectron2.modeling.roi_heads import fast_rcnn 
    fast_rcnn.fast_rcnn_inference_single_image = fast_rcnn_inference_single_image_all_scores    
    
    #evaluator = ATeacherTrainer.build_evaluator(cfg, cfg.DATASETS.TEST[0])
    dataloader = dataloaders[1]
    evaluator = build_evaluator(cfg, dataset_name=cfg.DATASETS.TEST[0], output_folder=os.path.join(cfg.OUTPUT_DIR, "DAS"))
    torch.set_grad_enabled(False)

    #res, preds_gallery = ATeacherTrainer.test(cfg, model, evaluators=[evaluator],
    #                                          data_loader=dataloader, perturb=False)
    #data_loader = ALDITrainer.build_test_loader(cfg, dataset)
    
    # Updated DAS - use forward hook to get preds_gallery
    preds_gallery = []
    forward_hook_handle = model.register_forward_hook(lambda module, input, output: preds_gallery.append(output))
    res = inference_on_dataset(model, dataloader, evaluator=evaluator)
    forward_hook_handle.remove()

    reg_list = [x[0]["instances"].get("pred_boxes").tensor for x in preds_gallery]
    #todo: should "scores" have logits applied.
    logits = [x[0]["instances"].get("scores_logits") for x in preds_gallery]    # Notion: Softmax before returning.

    results_bbox_per_img = []
    results_logits_per_img = []
    for reg in reg_list:
        bboxes_per_img = [reg[i].cpu().numpy() for i in range(len(reg))]
        results_bbox_per_img.append(bboxes_per_img)
    for logit in logits:
        logits_per_img = [logit[i].cpu().numpy() for i in range(len(logit))]
        results_logits_per_img.append(logits_per_img)

    scores_perModel = {}
    lambdas = [1] * max_repeat
    for key in lambdas:
        scores_perModel[key] = []
    for repeat_time in range(max_repeat):
        _lambda = lambdas[repeat_time]
        print("\nrepeat time: {}".format(repeat_time))
        model_copied = copy.deepcopy(model)
        # Updated DAS - apply perturbation to model rather than in model code
        model_copied = perturb_model_parameters(model_copied)

        #_, preds_gallery_perturb = ATeacherTrainer.test(cfg, model_copied, evaluators=[evaluator], 
        #                                                data_loader=dataloader, perturb=True)
        
        # Updated DAS - use forward hook for preds_gallery_perturb and perturb model prior
        preds_gallery_perturb = []        
        forward_hook_handle = model_copied.register_forward_hook(lambda module, input, output: preds_gallery_perturb.append(output))
        _ = inference_on_dataset(model_copied, dataloader, evaluator=evaluator)
        forward_hook_handle.remove()

        reg_list_perturbe = [x[0]["instances"].get("pred_boxes").tensor for x in preds_gallery_perturb]
        logits_perturbe = [x[0]["instances"].get("scores_logits") for x in preds_gallery_perturb]

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
        least_cost_final, iou_cost_v2, iou_cost_v1, kl_cost_v1, kl_cost_v2 = 0, 0, 0, 0, 0

        for img_idx in range(len(results_logits_per_img)):
            _bboxes = results_bbox_per_img[img_idx]
            _bboxes_perturbe = results_bbox_per_img_perturbe[img_idx]
            _logits = results_logits_per_img[img_idx]
            _logits_perturbe = results_logits_per_img_perturbe[img_idx]

            _bboxes = torch.Tensor(_bboxes)
            _bboxes_perturbe = torch.Tensor(_bboxes_perturbe)
            _logits = torch.Tensor(_logits)
            _logits_perturbe = torch.Tensor(_logits_perturbe)

            if len(_bboxes.shape) < 2 or len(_bboxes_perturbe.shape) < 2:
                continue
                
            

            
            sampleCnt, _ = _bboxes.shape
            sampleCntPerturb, _ = _bboxes_perturbe.shape
            max_match = min(sampleCnt, sampleCntPerturb)
            num_flag += 1
            
            match_quality_matrix = pairwise_iou(Boxes(_bboxes_perturbe), Boxes(_bboxes)) # gt rows, perturb cols
            match_row_iou, match_col_iou = linear_sum_assignment(match_quality_matrix, maximize=True)
            
            iou_cost = iou_loss(_bboxes_perturbe, _bboxes)
            kl_div = computeKLDivergenceMatrix(_logits_perturbe, _logits)

            costMatrix = iou_cost + kl_div * _lambda
            matched_rowIdx, matched_colIdx = linear_sum_assignment(costMatrix)
            least_cost_final += costMatrix[matched_rowIdx, matched_colIdx].sum().numpy().tolist() / max_match
            iou_cost_v1 += iou_cost[matched_rowIdx, matched_colIdx].sum().numpy().tolist() / max_match
            iou_cost_v2 += iou_cost[match_row_iou, match_col_iou].sum().numpy().tolist() / max_match
            kl_cost_v1 += kl_div[matched_rowIdx, matched_colIdx].sum().numpy().tolist() / max_match
            kl_cost_v2 += kl_div[match_row_iou, match_col_iou].sum().numpy().tolist() / max_match

        least_cost_final /= num_flag
        iou_cost_v2 /= num_flag
        iou_cost_v1 /= num_flag
        kl_cost_v1 /= num_flag
        kl_cost_v2 /= num_flag

        scores_perModel[_lambda].append(least_cost_final)
        print((scores_perModel))
        del model_copied

    ground_truth = res["bbox"]["AP50"]

    return {"score": scores_perModel,
            "ground_truth": ground_truth,
            "iou_cost_v2": iou_cost_v2,
            "iou_cost_v1": iou_cost_v1,
            "kl_cost_v2": kl_cost_v2,
            "kl_cost_v1": kl_cost_v1,}


def PDR(cfg, model, dataloaders, max_repeat):
    USE_BACKBONE_FEATURE = True
    SEMISUPNET_DIS_TYPE = ["p2","p3","p4","p5"] #todo: review which features should be used.
    assert len(dataloaders) == 2
    torch.set_grad_enabled(False)
    model.eval()

    dataloader_source, dataloader_target = dataloaders
    proto_target, _ = getPrototypes(model, dataloader_target,
                                                         USE_BACKBONE_FEATURE, SEMISUPNET_DIS_TYPE)
    proto_source, _ = getPrototypes(model, dataloader_source,
                                                         USE_BACKBONE_FEATURE, SEMISUPNET_DIS_TYPE)
    
    dist = calCateProtoDistance(proto_source, proto_target)
    
    return {"score": round(dist, 4),}


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
    div = round(((dist_crossDomain_diffCate * dist_sourceInDomain_diffCate * dist_targetInDomain_diffCate / dist_crossDomain_sameCate)).cpu().item(), 4)

    return div


def getPrototypes(model, dataloader, useBackboneFeature=True, featureNames=["vgg4"]):
    prototypes = []
    weights = []
    backboneFeatureList = []

    for idx, inputs in enumerate(dataloader):
        print("\robtaining prototypes: {}".format(idx), end="") if idx % 10 == 0 else ...
        assert not model.training

        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features, None)

        # results, _ = self.roi_heads(images, features, proposals, None)
        # In roi_heads.forward =>   
        # 
        #   pred_instances = self._forward_box(features, proposals)
        # In roi_heads._forward_box
        #         features = [features[f] for f in self.box_in_features]
        #         box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        #         box_features = self.box_head(box_features)
        #         predictions = self.box_predictor(box_features)
        features = [features[f] for f in featureNames] #todo: could do this without feature names.  Just take all the features returned.
        backboneFeatureList.append(torch.mean(features[0], dim=(2,3))) # Takes first feature name only. Not used later.

        box_features_backbone = model.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])  # [1000, 512, 7, 7]
        box_features_mlp = model.roi_heads.box_head(box_features_backbone)  # [1000, 1024]

        #todo: this needs to be fixed. Predictions is actual two returns including logits for all classes
        predictions = model.roi_heads.box_predictor(box_features_mlp) # FastRCNNOutputLayers.forward
        # self.box_predictor.inference(predictions, proposals)
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


class IoUCost():
    def __init__(self, iou_mode='iou', weight=1.): 
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bboxes, gt_bboxes):
        overlaps = _bbox_overlaps(bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)

        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight