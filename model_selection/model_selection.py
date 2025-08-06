import sys
import datetime
import json
import logging
import os
import pprint
import time
import copy
import numpy as np
from collections import defaultdict, OrderedDict
from typing import List, Mapping, Optional
import torch
from scipy.optimize import linear_sum_assignment
import pandas as pd

import torch.nn.functional as F
from torchvision.ops import complete_box_iou_loss
import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2.modeling import GeneralizedRCNN
from detectron2.data.build import filter_images_with_only_crowd_annotations, build_detection_train_loader, build_detection_test_loader
from detectron2.data.samplers import InferenceSampler
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import DefaultTrainer, PeriodicWriter, SimpleTrainer, HookBase, hooks
from detectron2.modeling import build_model
from detectron2.utils.logger import log_every_n_seconds, _log_api_usage
from fvcore.nn.giou_loss import giou_loss

from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.structures import pairwise_iou, Boxes, BoxMode
from fvcore.nn.giou_loss import giou_loss
from detectron2.evaluation import inference_on_dataset
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import pairwise_iou, Boxes
#from aldi.config import add_aldi_config
#from aldi.methodsDirectory2Fast import perturb_by_dropout, dropout_masks

from model_selection.utils import get_model_shortname, load_model_weights, build_evaluator, perturb_by_dropout, dropout_masks
from modelSeleTools_DAS.methodsDirectory2Fast import perturb_model_parameters

# Override box_loss methods to use mean
from .box_loss import _mean_dense_box_regression_loss, classifier_loss_on_gt_boxes, get_outputs_with_image_id
current_module = sys.modules['detectron2.modeling.proposal_generator.rpn']
setattr(current_module, '_dense_box_regression_loss', _mean_dense_box_regression_loss)
current_module = sys.modules['detectron2.modeling.roi_heads.fast_rcnn']
setattr(current_module, '_dense_box_regression_loss', _mean_dense_box_regression_loss)

from modelSeleTools_DAS.fast_rcnn import fast_rcnn_inference_single_image_all_scores

DEBUG = False
debug_dict = {}
logger = logging.getLogger("detectron2")


class ModelSelection:
    def __init__(self, cfg, source_ds=[], target_ds=None, gather_metric_period=1):
        self._hooks: List[HookBase] = []
        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int
        self.storage: EventStorage
        
        _log_api_usage("trainer." + self.__class__.__name__)    
        
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.cfg = self.update_cfg_loss(cfg)
        self.model = build_model(self.cfg)
        self.model.eval()
        self.perturb_method = perturb_by_dropout if cfg.MODEL_SELECTION.PERTURB_TYPE == "dropout" else perturb_model_parameters

        # Calculate dropout_masks so they can be used consistently throughout experiments
        if cfg.MODEL_SELECTION.PERTURB_TYPE == "dropout":
            self.perturbation_masks = [dropout_masks(self.model, p=cfg.MODEL_SELECTION.DROPOUT) for _ in range(cfg.MODEL_SELECTION.N_PERTURBATIONS)]
        else:
            self.perturbation_masks = None

        if cfg.MODEL_SELECTION.N_TRANSFORMED_SOURCE > 0:
            new_source_ds = []
            for i in range(cfg.MODEL_SELECTION.N_TRANSFORMED_SOURCE):
                new_source_ds += [f"{s_ds}_transformed{i}" for s_ds in source_ds]
            new_source_ds += source_ds
            source_ds = new_source_ds
        self.sampled_src = get_dataset_samples(source_ds, n=cfg.MODEL_SELECTION.N_SAMPLE) if source_ds is not None else []
        #self.sampled_src = self.sampled_src[:min(len(self.sampled_src), 6)]
        self.sampled_tgt = get_dataset_samples(target_ds, n=1000000) if target_ds is not None else [] # Use all target data

        self.evaluation_dir = os.path.join(cfg.OUTPUT_DIR, "model_selection")
        self.gather_metric_period = gather_metric_period
  
        
    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
        iter: Optional[int] = None,
    ) -> None:
        #logger = logging.getLogger(__name__)

        iter = self.iter if iter is None else iter
        if (iter + 1) % self.gather_metric_period == 0:
            try:
                SimpleTrainer.write_metrics(loss_dict, data_time, iter, prefix)
                self.storage.step()
            except Exception:
                logger.exception("Exception in writing metrics: ")
                raise

    def run_model_selection(self, model_weights, source=True, neptune_run=None):
        from detectron2.modeling.roi_heads import fast_rcnn 
        fast_rcnn.fast_rcnn_inference_single_image = fast_rcnn_inference_single_image_all_scores
        
        outputs = {}
        model_shortname = get_model_shortname(model_weights)
        sampled_datasets = self.sampled_src if source else self.sampled_tgt
        
        total = len(sampled_datasets)
        eta = 1000000000 # hack to get debug to work.
        info_freq = 5
        start_time = time.perf_counter()
        
        # Iterate through sampled datasets
        for ds_idx, dataset_name in enumerate(sampled_datasets):
            load_model_weights(model_weights, self.model)
            unique_dataset_name = f"src_{dataset_name}_{ds_idx:02}" if source else f"tgt_{dataset_name}_{ds_idx:02}"
            outputs[unique_dataset_name] = {'source': source}
            
            cfg = self.get_updated_cfg_for_model_selection(self.cfg, dataset_name, ds_idx)

            # og_test_dataset = DatasetCatalog.get(dataset_name)
            
            ### A) Generate Pseudo label for baseline for loss with perturbed model
            from detectron2.modeling.roi_heads import fast_rcnn 
            fast_rcnn.fast_rcnn_inference_single_image = fast_rcnn_inference_single_image_all_scores
            evaluator = build_evaluator(cfg, dataset_name, self.evaluation_dir, do_eval=True)
            forward_hook_returns = []
            forward_hook_handle = self.model.register_forward_hook(lambda module, inputs, outputs: forward_hook_returns.extend(get_outputs_with_image_id(inputs, outputs)))
            results = DefaultTrainer.test(self.cfg, self.model, evaluator) 
            forward_hook_handle.remove()
            
            if 'bbox' in results:
                outputs[unique_dataset_name].update(results['bbox'])  # Assumes bbox results
                logger.info(f"model_selection: Groundtruth for {dataset_name}, {model_shortname}: {pprint.pformat(results['bbox'])}")
                if neptune_run is not None:
                    for k, v in results['bbox'].items():
                        neptune_run[f"metrics/{model_shortname}/{unique_dataset_name}/{k}"] = v
                del evaluator

            dataset_file, pseudo_dataset_name = save_pseudo_coco_file(model_weights, self.evaluation_dir, dataset_name)
            register_coco_instances(pseudo_dataset_name, {}, dataset_file, "./")

            ### B) Calculate losses with perturbed model against pseudo label coco annotations         
            losses_perturbed = defaultdict(list)
            for n_perturb in range(self.cfg.MODEL_SELECTION.N_PERTURBATIONS):
                # Perturb model with dropout mask
                model_copied = copy.deepcopy(self.model)
                model_copied = self.perturb_method(model_copied, p=self.cfg.MODEL_SELECTION.DROPOUT, mask_dict=self.perturbation_masks, n=n_perturb)

                # B1) Calculate classification loss (classifier head loss) with perturbed model using pseudo gt boxes
                # Build data_loader from pseudo label dataset
                dataset = DatasetCatalog.get(pseudo_dataset_name)
                #data_loader = build_detection_train_loader(cfg, sampler=InferenceSampler(len(dataset))) # todo: need to use pseudo datasets
                data_loader = build_detection_test_loader(
                    dataset=dataset,
                    mapper=DatasetMapper(cfg, True),
                    sampler=InferenceSampler(len(dataset)),
                    #total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
                    #aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
                    num_workers=cfg.DATALOADER.NUM_WORKERS)
                # From FIS build_detection_test_loader(test_dataset, mapper=DatasetMapper(cfg, False), sampler=InferenceSampler(len(test_dataset)))    
                perturbed_predictions, perturbed_scores_logits = self.run_dataset(data_loader, model_copied) # Will calculate losses with pseudo label boxes
                    
                # B2) Calculate box loss with perturbed model compared to pseudo gt boxes
                # Generate perturbed boxes by running inference with perturbed model
                #perturbed_evaluation_dir = os.path.join(self.evaluation_dir, 'perturbed')
                #evaluator = build_evaluator(cfg, dataset_name, perturbed_evaluation_dir, do_eval=False)
                #cls_loss_returns = []
                # todo: pass in pred_instances from psuedo into hook
                #cls_loss_hook_handle = model_copied.roi_heads.register_forward_hook(
                #    lambda module, inputs, outputs: cls_loss_returns.append(classifier_loss_on_gt_boxes(module, inputs)))        
                #_ = DefaultTrainer.test(self.cfg, model_copied, evaluator) # use predicted boxes saved from this evaluation           
                #cls_loss_hook_handle.remove()
                
                # Now take the results and compare to pseudo gt boxes
                box_losses = calc_box_loss(forward_hook_returns, perturbed_predictions)
                losses_perturbed['loss_box_giou'].append(box_losses[0])
                losses_perturbed['loss_box_smooth_l1'].append(box_losses[1])
                losses_perturbed['loss_box_iou'].append(box_losses[2])
                
                kl_loss = calc_score_logits_loss(forward_hook_returns, perturbed_scores_logits)
                losses_perturbed['loss_score_logits'].append(kl_loss)

            # Average perturbed losses
            losses_perturbed = {k: (float(np.average(v)), float(np.std(v))) for k, v in losses_perturbed.items() if
                                "loss" in k}

            # Calculate entropy losses
            entropy_loss, info_max_reg = calc_entropy_measures(forward_hook_returns)
            losses_perturbed['entropy']= entropy_loss
            losses_perturbed['info_max_reg'] = info_max_reg
            
            if neptune_run is not None:
                for k, v in losses_perturbed.items():
                    if "loss" in k:
                        neptune_run[f"metrics/{model_shortname}/{unique_dataset_name}/{k}_mean"] = v[0]
                        neptune_run[f"metrics/{model_shortname}/{unique_dataset_name}/{k}_std"] = v[1]
            logger.info(f"model_selection: Perturbed loss for {dataset_name}, {model_shortname}: {pprint.pformat(losses_perturbed)}")
            outputs[unique_dataset_name].update(losses_perturbed)
                
            iters_after_start = ds_idx + 1
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if ds_idx % info_freq == 0 or ds_idx == total - 1:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - ds_idx - 1)))
                logger.info(f"Model selection for {str(model_shortname)} done {ds_idx + 1}/{total} sampled datasets. Total for {cfg.MODEL_SELECTION.N_PERTURBATIONS} perturbations: {total_seconds_per_iter:.4f} s/iter. ETA={eta}")
        return outputs


    def run_dataset(self, data_loader, model):
        # Run calculations over all of dataset
        logger.info("model_selection: Start calculations on dataset")
    
        #model.training = True
        #model.train() # put it in train so losses are calculated #todo: might be able to remove this
        
        # Create hook on model.roi_heads._forward_box(features, proposals)
        cls_loss_returns = []
        # todo: pass in pred_instances from psuedo into hook
        cls_loss_hook_handle = model.roi_heads.register_forward_hook(
            lambda module, inputs, outputs: cls_loss_returns.append(classifier_loss_on_gt_boxes(module, inputs)))        
        start_iter = 0
        self.iter = start_iter
        self.storage = EventStorage()
        predictions = []
        with EventStorage(start_iter) as self.storage:
            try:
                
                total = len(data_loader.dataset) // data_loader.batch_size
                num_warmup = min(5, total - 1)
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                start_data_time = time.perf_counter()
                
                for idx, inputs in enumerate(data_loader):
                    total_data_time += time.perf_counter() - start_data_time
                    if idx == num_warmup:
                        start_time = time.perf_counter()
                        total_data_time = 0
                        total_compute_time = 0    
                    start_compute_time = time.perf_counter()
                    
                    outputs = self.run_step(inputs, model)
                    # Based on inference_on_dataset and coco_evaluation.process
                    for input, output in zip(inputs, outputs):
                        prediction = {"image_id": input["image_id"]}
                        if "instances" in output:
                            instances = output["instances"].to("cpu")
                            #prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
                            prediction["instances"] = instances # Boxes (XYXY)
                            predictions.append(prediction)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    total_compute_time += time.perf_counter() - start_compute_time
            
                    iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                    data_seconds_per_iter = total_data_time / iters_after_start
                    compute_seconds_per_iter = total_compute_time / iters_after_start
                    total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                    if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                        eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                        log_every_n_seconds(
                            logging.INFO,
                            (
                                f"Inference done {idx + 1}/{total}. "
                                f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                                f"Calculation: {compute_seconds_per_iter:.4f} s/iter. "
                                f"Total: {total_seconds_per_iter:.4f} s/iter. "
                                f"ETA={eta}"
                            ),
                            name="detectron2",
                            n=5,
                        )
                    start_data_time = time.perf_counter()
                    self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
        loss_cls_on_gt_boxes = [l[0] for l in cls_loss_returns]
        scores_logits = [l[1] for l in cls_loss_returns]
        cls_loss_hook_handle.remove()
        #losses_avg = {k: v._global_avg for k, v in self.storage.histories().items() if 'loss' in k}
        #losses_avg['loss_cls_gt_boxes'] = np.mean(np.array(loss_cls_on_gt_boxes))
        #losses_avg['scores_logits'] =scores_logits
        #return losses_avg
        return predictions, scores_logits
    
    
    def run_step(self, data, model):
        start = time.perf_counter()
        #data_time = time.perf_counter() - start
        #model.training = True # so that it calculates losses
        #model.train() 
        with torch.no_grad():
            outputs = inference_with_targets(model, data) # model in training so targets are used
        #self._write_metrics(losses, data_time) # Collates loss_dict
        return outputs

    def run_step_inference(self, data, model):
        assert model.training, "[ModelSelection] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[ModelSelection] CUDA is required for training!"

        start = time.perf_counter()
        data_time = time.perf_counter() - start
        outputs = model(data)

        if isinstance(loss_dict, torch.Tensor):
            loss_dict = {"total_loss": loss_dict}

        # Match boxes using Faster-RCNN matching quality (from detectron2.modelling.proposal_generator.rpn)
        match_quality_matrix = pairwise_iou(outputs.gt_boxes, data.targets)  # gt rows, perturb cols
        match_row, match_col = linear_sum_assignment(match_quality_matrix, maximize=True)

        # From BoS - which doesn't use perturb but dropout
        #iou_cost_final = iou_cost
        #iou_matched_row_inds, iou_matched_col_inds = linear_sum_assignment(iou_cost_final)
        #least_iou_cost_final = iou_cost_final[match_row, match_col].sum().numpy().tolist() / max_match

        # Mine
        giou_cost = giou_loss(_bboxes_perturbe[match_row, :], _bboxes[match_col, :], reduction="mean").item()

        loss_dict = {'loss_box_reg_giou': giou_cost}
        self._write_metrics(loss_dict, data_time)



    def get_updated_cfg_for_model_selection(self, cfg, dataset_name, seed_offset):
        # cfg = setup(args)
        cfg.defrost()
        cfg.SEED = cfg.SEED + seed_offset  # update so new random selection
        
        cfg.DATASETS.TRAIN = (dataset_name,)
        cfg.DATASETS.TEST = cfg.DATASETS.TRAIN
        
        # We're using training to calculate losses but only want the same aug as TEST.
        cfg.INPUT.MIN_SIZE_TRAIN = cfg.INPUT.MIN_SIZE_TEST
        cfg.INPUT.MAX_SIZE_TRAIN = cfg.INPUT.MAX_SIZE_TEST
        cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'choice' 
        cfg.INPUT.RANDOM_FLIP = "none"
        cfg.freeze()
        return cfg


    def update_cfg_loss(self, cfg):
        cfg.defrost()
        #vanilla_cfg = get_cfg()
        #add_aldi_config(vanilla_cfg)
        #cfg.AUG = vanilla_cfg.AUG
        #cfg.DOMAIN_ADAPT = vanilla_cfg.DOMAIN_ADAPT
        #cfg.EMA = vanilla_cfg.EMA
        cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "giou"
        cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "giou"
        cfg.SOLVER.IMS_PER_BATCH = 1
        #cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = True
        #cfg.DATASETS.TEST = (dataset_name,)
        #cfg.DATASETS.TRAIN = (dataset_name,)
        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
        #cfg.DATASETS.BATCH_CONTENTS = ("labeled_weak",)
        #cfg.DATASETS.BATCH_RATIOS = (1,)# Will apply weak augmentation
        #cfg.SOLVER.MAX_ITER = 1
        #cfg.TEST.EVAL_PERIOD = 1
        #cfg.EMA.ENABLED = False
        cfg.freeze()
        return cfg
    
def inference_with_targets(
    model,
    batched_inputs,
    do_postprocess: bool = True,
):
    """
    Run inference on the given inputs.

    Args:
        batched_inputs (list[dict]): same as in :meth:`forward`
        do_postprocess (bool): whether to apply post-processing on the outputs.

    Returns:
        When do_postprocess=True, same as in :meth:`forward`.
        Otherwise, a list[Instances] containing raw network outputs.
    """
    images = model.preprocess_image(batched_inputs)
    features = model.backbone(images.tensor)

    if model.proposal_generator is not None:
        proposals, _ = model.proposal_generator(images, features, None)
    else:
        assert "proposals" in batched_inputs[0]
        proposals = [x["proposals"].to(model.device) for x in batched_inputs]

    if "instances" in batched_inputs[0]:
        gt_instances = [x["instances"].to(model.device) for x in batched_inputs]
    else:
        gt_instances = None
    results, _ = model.roi_heads(images, features, proposals, gt_instances)

    if do_postprocess:
        assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
        return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
    return results

    
def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        results.append(result)
    return results


def save_pseudo_coco_file(model_weights, results_path, dataset_name, score_threshold=0.0):
    # use the results from a test of the dataset and the original dataset
    # image list to create a new coco format file of pseudo labels.
    # Assumes results from last test were saved to inference/coco_instances_results.json
    
    dataset_images = DatasetCatalog.get(dataset_name)
    dataset_metadata = MetadataCatalog.get(dataset_name)
    
    # Remove annotations which are the ground truth
    for d in dataset_images:
        d["id"] = d["image_id"]
        del d["image_id"]
        del d['annotations']
        
    # Get annotations from results from last test run  
    coco_results_file = os.path.join(results_path, "coco_instances_results.json")
    with open(coco_results_file, 'r') as file:
        logger.info(f"model_selection: Loading results from {coco_results_file}")
        psuedo_annotations = json.load(file)
    ann_by_image = defaultdict(list)
    for ann in psuedo_annotations:
        ann_by_image[ann['image_id']].append(ann)
    annotations = []
    
    ann_id = 0
    for img_id, anns in ann_by_image.items():
        for ann in anns:
            if ann["score"] > score_threshold:
                bbox = [int(p) for p in ann["bbox"]]
                coco_ann = {
                    'category_id': ann["category_id"],
                    'image_id': ann["image_id"],
                    'bbox': bbox,
                    'id': ann_id
                }
                annotations.append(coco_ann)
                ann_id += 1

    info = {
        'model': model_weights,
        'dataset': dataset_name,
        'description': f"Pseudo labels generated from model {model_weights} on dataset {dataset_name}",
    }
    
    # unmap the category mapping ids for COCO
    if hasattr(dataset_metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in dataset_metadata.thing_dataset_id_to_contiguous_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa
    categories = [
        {"id": reverse_id_mapper(idx), "name": name}
        for idx, name in enumerate(dataset_metadata.thing_classes)
    ]
    psuedo_coco_dataset = {
        'info': info,
        'images': dataset_images,
        'annotations': annotations,
        'categories': categories,
        'licenses': None}
    
    # Generate names related to model and dataset used
    model_weights = 'zz'.join(model_weights[:-4].split('/')[1:]) #todo: make more robust
    model_dataset_name = f"{model_weights}__{dataset_name}"
    dataset_file = os.path.join(results_path, f"psuedo_{model_dataset_name}.json")

    with open(dataset_file, "w") as fp:
        json.dump(psuedo_coco_dataset, fp)
    logger.info(f"model_selection: Saving new coco file with psuedo labels from dataset {dataset_name} and model weights {model_weights} at {dataset_file}")
    return dataset_file, model_dataset_name


def calc_box_loss(gt_instances, predicted_instances):
    # compare results and calc box loss
    logger.info("model_selection: Calculating box loss")
    #dataset_images = DatasetCatalog.get(dataset_name) # psuedo dataset
    #dataset_metadata = MetadataCatalog.get(dataset_name)
    # unmap the category mapping ids for COCO
    #if hasattr(dataset_metadata, "thing_dataset_id_to_contiguous_id"):
    #    reverse_id_mapping = {v: k for k, v in dataset_metadata.thing_dataset_id_to_contiguous_id.items()}
    #    reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    #else:
    #    reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa    
        
    # Remove annotations which are the ground truth
    #for d in dataset_images:
    #    d["id"] = d["image_id"]
    #    del d["image_id"]
    #    del d['annotations']
        
    # Get annotations from results from last test run  
    #coco_results_file = os.path.join(perturbed_results_dir, "coco_instances_results.json")
    #with open(coco_results_file, 'r') as file:
    #    logger.info(f"model_selection: Calculating box loss - Loading results from {coco_results_file}")
    #    perturbed_annotations = json.load(file) # perturbed results
    #perturbed_ann_by_image = defaultdict(list)
    #for ann in perturbed_annotations:
        #if ann['score'] > threshold:
    #    ann['category_id'] = dataset_metadata.thing_dataset_id_to_contiguous_id[ann['category_id']]
    #    perturbed_ann_by_image[ann['image_id']].append(ann)
    
    #match_quality = 0
    ious, gious, smooth_l1_losses = [], [], []
    no_matches = []
    for gt, pred in zip(gt_instances, predicted_instances):        
        height, width = gt["instances"].image_size
        scale_x, scale_y = (
            1 / width,
            1 / height,
        )
        scaling = np.array([scale_x, scale_y, scale_x, scale_y])
        
        #perturbed_anns = perturbed_ann_by_image[instance["image_id"]]
        #perturbed_boxes = np.array([BoxMode.convert(a["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for a in perturbed_anns])
        pred_boxes = np.array(pred["instances"].pred_boxes)        
        if pred_boxes.shape[0] > 0:
            pred_boxes = pred_boxes * scaling
        #pseudo_anns = instance['annotations']
        #pseudo_boxes = np.array([BoxMode.convert(a["bbox"], a["bbox_mode"], BoxMode.XYXY_ABS) for a in pseudo_anns])
        gt_boxes = np.array(gt["instances"].pred_boxes.to("cpu"))
        if len(gt_boxes) > 0:
            gt_boxes = gt_boxes * scaling
        
        #max_match = min(len(perturbed_anns), len(pseudo_anns))
        # Match boxes using Faster-RCNN matching quality (from detectron2.modelling.proposal_generator.rpn)
        pairwise_iou_matrix = pairwise_iou(Boxes(pred_boxes), Boxes(gt_boxes)) # gt rows, perturb cols
        match_row_pairwise_iou, match_col_pairwise_iou = linear_sum_assignment(pairwise_iou_matrix, maximize=True)
        if len(match_row_pairwise_iou) == 0:
            no_matches.append(pred["image_id"])
            continue
        iou = pairwise_iou_matrix[match_row_pairwise_iou, match_col_pairwise_iou].mean().item()
        giou = 1-giou_loss(torch.Tensor(pred_boxes[match_row_pairwise_iou,:]), torch.Tensor(gt_boxes[match_col_pairwise_iou, :]), reduction="mean").item() 
        smooth_l1_l = F.smooth_l1_loss(torch.Tensor(pred_boxes[match_row_pairwise_iou,:]), torch.Tensor(gt_boxes[match_col_pairwise_iou, :]), reduction="mean").item()
        ious.append(iou)
        gious.append(giou)
        smooth_l1_losses.append(smooth_l1_l)
    logger.info(f"model_selection: No matched boxes found for image_ids {no_matches}")
    return np.mean(gious), np.mean(smooth_l1_losses), np.mean(ious)


def calc_score_logits_loss(gt_instances, predicted_scores):
    logger.info("Calculating KL divergence loss")
    kl_loss_calc = torch.nn.KLDivLoss(reduction="batchmean")
    gt_scores, pred_scores = [], []
    for i, gt in enumerate(gt_instances):
        gt_score = gt["instances"].scores_logits
        pred_score = torch.Tensor(predicted_scores[i]).to(gt_score.device)
        if gt_score.shape != pred_score.shape or gt_score.shape[0] == 0:
            logger.warning(f"model_selection: gt logit and perturbed logit shapes don't match for image {i} or gt is empty - gt:{gt_score.shape}, pred:{pred_score.shape}")
        else:
            gt_scores.append(gt_score)
            pred_scores.append(pred_score)
    gt_scores = torch.vstack(gt_scores)
    pred_scores = torch.vstack(pred_scores)    
    # Compute KL divergence loss
    kl_losses = kl_loss_calc(pred_scores, gt_scores)
    return kl_losses.item()


def calc_entropy_measures(gt_instances):
    logger.info("model_selection: Calculating entropy measures")
    gt_scores = [gt["instances"].scores_logits for gt in gt_instances]
    gt_scores = torch.vstack(gt_scores)
    entropy = torch.mean(torch.sum(torch.log(gt_scores + 1e-7) * gt_scores, dim=1)) # sum across class then mean
    gt_scores_averaged = torch.mean(gt_scores, dim=0)
    info_max_reg = torch.sum(torch.log(gt_scores_averaged + 1e-7)*gt_scores_averaged)
    return entropy.item(), info_max_reg.item()


def register_coco_instances_with_split(name, parent, json_file, image_root, indices, filter_empty):
    if name in DatasetCatalog.keys():
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)
    DatasetCatalog.register(name,
                            lambda: load_coco_json_with_split(json_file, image_root, parent, indices, filter_empty))
    metadata = MetadataCatalog.get(parent)
    MetadataCatalog.get(name).set(json_file=json_file, image_root=image_root, name=name,
                                  thing_classes=metadata.thing_classes,
                                  thing_dataset_id_to_contiguous_id=metadata.thing_dataset_id_to_contiguous_id)


def load_coco_json_with_split(json_file, image_root, parent_name, indices, filter_empty):
    dataset_dicts = load_coco_json(json_file, image_root, parent_name)
    if filter_empty:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    logger.info("model_selection: Splitting off {} images of {} images in COCO format from {}: first 10 indices {}".format(
        len(indices), len(dataset_dicts), json_file, indices[:min(len(indices), 10)]))
    return [dataset_dicts[index] for index in indices]


def split_dataset_into_samples(dataset_name, sample_size, seed=0):
    # get metadata
    metadata = MetadataCatalog.get(dataset_name)

    # get dataset and split indices
    dataset_dicts = DatasetCatalog.get(dataset_name)
    num_instances = len(dataset_dicts)
    indices = np.arange(num_instances)
    np.random.seed(seed)
    np.random.shuffle(indices)
    new_ds = []
    for i in range(num_instances // sample_size):
        new_ds_indices = sorted(indices[i * sample_size:(i + 1) * sample_size])
        new_ds_name = f"{dataset_name}_sample_{i:03}"
        new_ds.append(new_ds_name)
        # register datasets
        register_coco_instances_with_split(new_ds_name, metadata.name, metadata.json_file, metadata.image_root,
                                           new_ds_indices, filter_empty=False)
        logger.info(f"model_selection: Registered coco dataset with split: {new_ds_name}")
    new_ds = [dataset_name] if len(new_ds) == 0 else new_ds
    return new_ds


def get_dataset_samples(dataset_names, n=250):
    samples = []
    for ds in dataset_names:
        samples = samples + split_dataset_into_samples(ds, sample_size=n)
    return samples





