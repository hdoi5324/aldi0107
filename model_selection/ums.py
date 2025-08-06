import copy
import logging
import os

import numpy as np
import torch
from detectron2.data.build import build_detection_test_loader
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import InferenceSampler

from detectron2.evaluation import inference_on_dataset
from detectron2.structures import pairwise_iou
from fvcore.nn.giou_loss import giou_loss
from scipy.optimize import linear_sum_assignment

from model_selection.fast_rcnn import fast_rcnn_inference_single_image_all_scores
from model_selection.utils import build_evaluator, perturb_by_dropout

# Override box_loss methods to use mean todo: remove this stuff
# from .box_loss import _mean_dense_box_regression_loss, classifier_loss_on_gt_boxes, get_outputs_with_image_id
# current_module = sys.modules['detectron2.modeling.proposal_generator.rpn']
# setattr(current_module, '_dense_box_regression_loss', _mean_dense_box_regression_loss)
# current_module = sys.modules['detectron2.modeling.roi_heads.fast_rcnn']
# setattr(current_module, '_dense_box_regression_loss', _mean_dense_box_regression_loss)

DEBUG = False
debug_dict = {}
logger = logging.getLogger("detectron2")

def test_ums(cfg, model, perturbation_types=['das'], evaluation_dir="ums"):
    dataloader = get_ums_dataloader(cfg.UMS.UNLABELED, cfg)
    evaluation_dir = os.path.join(cfg.OUTPUT_DIR, evaluation_dir)
    evaluator = build_evaluator(cfg, cfg.UMS.UNLABELED, evaluation_dir, do_eval=True)
    ums_calculator = UMS(cfg, model, dataloader, evaluator, perturbation_types=perturbation_types)
    return ums_calculator.calculate_measures()
    

def get_ums_dataloader(unlabeled_dataset, cfg):
    dataset = DatasetCatalog.get(unlabeled_dataset)
    data_loader = build_detection_test_loader(
        dataset=dataset,
        mapper=DatasetMapper(cfg, False),
        sampler=InferenceSampler(len(dataset)),
        num_workers=cfg.DATALOADER.NUM_WORKERS)
    return data_loader


class UMS:
    def __init__(self, cfg, model, dataloader, evaluator, measures=['iou'], perturbation_types=['das']):
        self.cfg = cfg.clone()
        self.model = model
        self.dataloader = dataloader
        self.evaluator = evaluator
        self.measures = measures
        

    def calculate_measures(self, scale_boxes=True, layers=[5]):
        all_results = {}
        from detectron2.modeling.roi_heads import fast_rcnn 
        fast_rcnn.fast_rcnn_inference_single_image = fast_rcnn_inference_single_image_all_scores
        self.model.eval()
        
        # Unperturbed forward hook
        unperturbed_returns = []
        forward_hook_handle = self.model.register_forward_hook(lambda module, inputs, outputs: unperturbed_returns.extend(get_outputs_with_image_id(inputs, outputs)))
        results = inference_on_dataset(self.model, self.dataloader, self.evaluator)
        forward_hook_handle.remove()
        if scale_boxes:
            self.scale_boxes(unperturbed_returns)

        # Perturbed forward hook - dropout
        layer_combinations = [[5], [2,3,4,5]]
        for layers in layer_combinations:
            model_copied = copy.deepcopy(self.model)
            model_copied = perturb_by_dropout(model_copied, p=self.cfg.UMS.DROPOUT, layer_nos=layers)
            perturbed_returns = []
            forward_hook_handle = model_copied.register_forward_hook(lambda module, inputs, outputs: perturbed_returns.extend(get_outputs_with_image_id(inputs, outputs)))
            _ = inference_on_dataset(model_copied, self.dataloader, self.evaluator)
            forward_hook_handle.remove() 
            del model_copied
            if scale_boxes:
                self.scale_boxes(perturbed_returns)
            results_dict_dropout = calc_ums_measures(unperturbed_returns, perturbed_returns)
            all_results[f"ums_{'_'.join([str(l) for l in layers])}"] = results_dict_dropout
        
        # Perturbed forward hook - DAS paramater perturbation
        model_copied = copy.deepcopy(self.model)
        model_copied = perturb_model_parameters(model_copied)
        perturbed_returns = []
        forward_hook_handle = model_copied.register_forward_hook(lambda module, inputs, outputs: perturbed_returns.extend(get_outputs_with_image_id(inputs, outputs)))
        _ = inference_on_dataset(model_copied, self.dataloader, self.evaluator)
        forward_hook_handle.remove()
        del model_copied
        if scale_boxes:
            self.scale_boxes(perturbed_returns)
        results_dict_das = calc_ums_measures(unperturbed_returns, perturbed_returns)        

        self.model.train()
        all_results["umsdas"] = results_dict_das
        all_results["gt"] = results
        return all_results
    
    
    def scale_boxes(self, forward_results):
        for r in forward_results:        
            height, width = r["instances"].image_size
            scale_x, scale_y = (
                1 / width,
                1 / height,
            )
            pred_boxes = r["instances"].pred_boxes
            if len(pred_boxes) > 0:
                pred_boxes.scale(scale_x, scale_y)


def get_outputs_with_image_id(inputs, outputs):
    for i, o in enumerate(outputs):
        o["image_id"] = inputs[0][i]['image_id']
    return outputs     

def calc_ums_measures(unperturbed_results, perturbed_results):
    logger.info("model_selection: Calculating box loss")
    measures=["iou", "giou", "kldiv", "ioukl", "ioukl_iou", "ioukl_kl"]
    measure_calc = {m: [] for m in measures}

    #ious, gious, smooth_l1_losses = [], [], []
    no_matches = []
    measure_results = {}
    # Iterate through images
    for gt, pred in zip(unperturbed_results, perturbed_results):        
        pred_boxes = pred["instances"].pred_boxes
        pred_scores = pred["instances"].scores_logits
        gt_boxes = gt["instances"].pred_boxes
        gt_scores = gt["instances"].scores_logits
        
        def calc_mean_match(matrix, maximize=True):
            match_row, match_col = linear_sum_assignment(matrix, maximize=maximize)
            if len(match_row) == 0:
                return None, None, None
            calc_mean = matrix[match_row, match_col].mean().item()
            return calc_mean, match_row, match_col
        
        # calculate matrix, use bipartite matching to find best matched boxes and calculate mean
        iou_matrix = pairwise_iou(pred_boxes, gt_boxes).to("cpu") # gt rows, perturb cols
        iou, match_row_iou, match_col_iou = calc_mean_match(iou_matrix)
        if iou is not None:
            measure_calc["iou"].append(iou)
        
            # Use iou bipartitie matching of boxes for giou calc.
            giou = 1-giou_loss(torch.Tensor(pred_boxes.tensor[match_row_iou,:]), torch.Tensor(gt_boxes.tensor[match_col_iou, :]), reduction="mean").item() 
            measure_calc["giou"].append(float(giou))
        
        kl_div_matrix = computeKLDivergenceMatrix(pred_scores, gt_scores).to("cpu")
        kl_div, _, _ = calc_mean_match(kl_div_matrix, maximize=False) # loss so minimize
        if kl_div is not None:
            measure_calc["kldiv"].append(kl_div)
        
        ioukl_matrix = iou_matrix - kl_div_matrix
        ioukl, match_row_ioukl, match_col_ioukl = calc_mean_match(ioukl_matrix)
        if ioukl is not None:
            measure_calc["ioukl"].append(ioukl)
            measure_calc["ioukl_iou"].append(float(iou_matrix[match_row_ioukl, match_col_ioukl].mean().item()))
            measure_calc["ioukl_kl"].append(-1.0*float(kl_div_matrix[match_row_ioukl, match_col_ioukl].mean().item()))
    measure_results = {m: float(np.mean(k)) for m, k in measure_calc.items()}
    
    # Calculate entropy losses
    entropy_loss, info_max_reg = calc_entropy_measures(unperturbed_results)
    measure_results['entropy']= float(entropy_loss)
    measure_results['info_max_reg'] = float(info_max_reg)
    
    if len(no_matches) > 0:
        logger.info(f"model_selection: No matched boxes found for image_ids {no_matches}")
    return measure_results


def computeKLDivergenceMatrix(logits1: torch.Tensor, logits2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    log_probs1 = logits1.log().unsqueeze(1)
    log_probs2 = logits2.log().unsqueeze(0)
    
    kl_matrix = torch.sum(logits1.unsqueeze(1) * (log_probs1 - log_probs2), dim=-1)

    return kl_matrix


def calc_entropy_measures(gt_instances):
    logger.info("model_selection: Calculating entropy measures")
    gt_scores = [gt["instances"].scores_logits for gt in gt_instances]
    gt_scores = torch.vstack(gt_scores)
    entropy = torch.mean(torch.sum(torch.log(gt_scores + 1e-7) * gt_scores, dim=1)) # sum across class then mean
    gt_scores_averaged = torch.mean(gt_scores, dim=0)
    info_max_reg = torch.sum(torch.log(gt_scores_averaged + 1e-7)*gt_scores_averaged)
    return entropy.item(), info_max_reg.item()


def perturb_model_parameters(module, **kwargs):
    # Based on DAS.  rcnn.py in DAobjTwoStagePseudoLabGeneralizedRCNN.inference method.
    # Applies perturbation to model rather than in the model class.
    
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
