import sys
import datetime
import json
import pprint
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

import torch.nn.functional as F
from torchvision.ops import complete_box_iou_loss
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.build import filter_images_with_only_crowd_annotations, build_detection_train_loader
from detectron2.data.samplers import InferenceSampler
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.engine import DefaultTrainer, PeriodicWriter, SimpleTrainer, HookBase, hooks
from detectron2.evaluation import DatasetEvaluators
from detectron2.modeling import build_model
from detectron2.utils.logger import log_every_n_seconds, _log_api_usage
from fvcore.common.checkpoint import _IncompatibleKeys
from fvcore.nn.giou_loss import giou_loss

from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.structures import pairwise_iou, Boxes, BoxMode

from aldi.config import add_aldi_config
from aldi.evaluation import Detectron2COCOEvaluatorAdapter
from aldi.methodsDirectory2Fast import perturb_by_dropout, dropout_masks

# Override box_loss methods to use mean
from .box_loss import _mean_dense_box_regression_loss, classifier_loss_on_gt_boxes
current_module = sys.modules['detectron2.modeling.proposal_generator.rpn']
setattr(current_module, '_dense_box_regression_loss', _mean_dense_box_regression_loss)
current_module = sys.modules['detectron2.modeling.roi_heads.fast_rcnn']
setattr(current_module, '_dense_box_regression_loss', _mean_dense_box_regression_loss)

DEBUG = False
debug_dict = {}
logger = logging.getLogger("detectron2")


class ModelSelection:
    def __init__(self, cfg, source_ds, target_ds=None, n_samples=250, dropout=0.1, n_perturbations=3, gather_metric_period=1):
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

        # Calculate dropout_masks so they can be used consistently throughout experiments
        self.perturbation_masks = [dropout_masks(self.model, p=dropout) for _ in range(n_perturbations)]

        self.sampled_src = get_dataset_samples(source_ds, n=n_samples)
        self.sampled_tgt = get_dataset_samples(target_ds, n=n_samples) if target_ds is not None else []

        self.evaluation_dir = os.path.join(cfg.OUTPUT_DIR, "model_selection")
        self.gather_metric_period = gather_metric_period
        
    def model_shortname(self, model_weights):
        model_weights[-17:-4] #todo: make more robust


    def load_model_weights(self, path):
        checkpointer = DetectionCheckpointer(self.model)  
        ret = checkpointer.load(path)

        if path.endswith(".pth") and "ema" in ret.keys():
            # self.logger.info("Loading EMA weights as model starting point.")
            ema_dict = {
                k.replace('model.', ''): v for k, v in ret['ema'].items()
            }
            # incompatible = self.model.load_state_dict(ema_dict, strict=False)
            ret['model'] = ema_dict
            incompatible = checkpointer._load_model(ret)
            if incompatible is not None:
                checkpointer._log_incompatible_keys(_IncompatibleKeys(
                    missing_keys=incompatible.missing_keys,
                    unexpected_keys=incompatible.unexpected_keys,
                    incorrect_shapes=[]
                ))
        return ret

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Just do COCO Evaluation."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "model_selection")
        evaluator = DatasetEvaluators([Detectron2COCOEvaluatorAdapter(dataset_name, output_dir=output_folder)])
        # Update evaluator to only check sampled images if it's a sampled dataset
        if "_sample_" in dataset_name:
            img_keys = [d['image_id'] for d in DatasetCatalog.get(dataset_name)]
            evaluator._evaluators[0]._coco_api.imgs = {k: v for k, v in
                                                       evaluator._evaluators[0]._coco_api.imgs.items() if
                                                       k in img_keys}
        return evaluator

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
        ]

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret    
        
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
        outputs = {}
        model_shortname = self.model_shortname(model_weights)
        sampled_datasets = self.sampled_src if source else self.sampled_tgt
        
        # Iterate through sampled datasets
        for ds_idx, dataset_name in enumerate(sampled_datasets):
            self.load_model_weights(model_weights)
            unique_dataset_name = f"src_{dataset_name}_{ds_idx:02}" if source else f"tgt_{dataset_name}_{ds_idx:02}"
            outputs[unique_dataset_name] = {'source': source}
            
            cfg = self.get_updated_cfg_for_model_selection(self.cfg, dataset_name, ds_idx)

            # og_test_dataset = DatasetCatalog.get(dataset_name)
            
            ### A) Generate Pseudo label for baseline for loss with perturbed model
            evaluator = ModelSelection.build_evaluator(cfg, dataset_name, self.evaluation_dir)
            results = DefaultTrainer.test(self.cfg, self.model, evaluator)
            
            outputs[unique_dataset_name].update(results['bbox'])  # Assumes bbox results
            logger.info(f"model_selection: Groundtruth for {dataset_name}, {model_shortname}: {pprint.pformat(results['bbox'])}")
            if neptune_run is not None:
                for k, v in results['bbox'].items():
                    neptune_run[f"metrics/{model_shortname}/{unique_dataset_name}/{k}"] = v
            del evaluator

            dataset_file, pseudo_dataset_name = save_pseudo_coco_file(model_weights, self.evaluation_dir, dataset_name,
                                                                      cfg.MODEL_SELECTION.SCORE_THRESHOLD)
            register_coco_instances(pseudo_dataset_name, {}, dataset_file, "./")

            ### B) Calculate losses with perturbed model against pseudo label coco annotations         
            losses_perturbed = defaultdict(list)
            for n_perturb in range(len(self.perturbation_masks)):
                # Build data_loader
                dataset = DatasetCatalog.get(dataset_name)
                data_loader = build_detection_train_loader(cfg, sampler=InferenceSampler(len(dataset)))
                # Perturb model with dropout mask
                model_copied = copy.deepcopy(self.model)
                model_copied = perturb_by_dropout(model_copied, p=cfg.MODEL_SELECTION.DROPOUT, mask_dict=self.perturbation_masks[n_perturb])

                # B1) Calculate classifier head loss with perturbed model using pseudo gt boxes
                losses = self.run_dataset(data_loader, model_copied)
                for k in losses.keys():
                    losses_perturbed[k].append(losses[k])
                    
                # B2) Calculate box loss with perturbed model compared to pseudo gt boxes
                # Generate perturbed boxes by running inference with perturbed model
                perturbed_evaluation_dir = os.path.join(self.evaluation_dir, 'perturbed')
                evaluator = ModelSelection.build_evaluator(cfg, dataset_name, perturbed_evaluation_dir)
                _ = DefaultTrainer.test(self.cfg, model_copied, evaluator)
                
                # Now take the results and compare to pseudo gt boxes
                box_losses = calc_box_loss(pseudo_dataset_name, perturbed_evaluation_dir, threshold=cfg.MODEL_SELECTION.SCORE_THRESHOLD)
                losses_perturbed['loss_box_giou'].append(box_losses[0])
                losses_perturbed['loss_box_smooth_l1'].append(box_losses[1])
                losses_perturbed['loss_box_iou'].append(box_losses[2])
                
            losses_perturbed = {k: (float(np.average(v)), float(np.std(v))) for k, v in losses_perturbed.items() if
                                "loss" in k}
            if neptune_run is not None:
                for k, v in losses_perturbed.items():
                    if "loss" in k:
                        neptune_run[f"metrics/{model_shortname}/{unique_dataset_name}/{k}_mean"] = v[0]
                        neptune_run[f"metrics/{model_shortname}/{unique_dataset_name}/{k}_std"] = v[1]
            logger.info(f"model_selection: Perturbed loss for {dataset_name}, {model_shortname}: {pprint.pformat(losses_perturbed)}")
            outputs[unique_dataset_name].update(losses_perturbed)
            
        return outputs


    def run_dataset(self, data_loader, model):
        # Run calculations over all of dataset
        logger.info("model_selection: Start calculations on dataset")
    
        model.training = True
        model.train() # put it in train so losses are calculated #todo: might be able to remove this
        
        # Create hook on model.roi_heads._forward_box(features, proposals)
        loss_cls_on_gt_boxes = []
        cls_loss_hook_handle = model.roi_heads.register_forward_hook(
            lambda module, inputs, outputs: loss_cls_on_gt_boxes.append(classifier_loss_on_gt_boxes(module, inputs)))        
        start_iter = 0
        self.iter = start_iter
        self.storage = EventStorage()
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
                            n=5,
                        )
                    start_data_time = time.perf_counter()
                    self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
        cls_loss_hook_handle.remove()
        losses_avg = {k: v._global_avg for k, v in self.storage.histories().items() if 'loss' in k}
        losses_avg['loss_cls_gt_boxes'] = np.mean(np.array(loss_cls_on_gt_boxes))
        return losses_avg
    
    
    def run_step(self, data, model):
        start = time.perf_counter()
        data_time = time.perf_counter() - start
        model.training = True
        model.train() 
        losses = model(data) # model in training so targets are used
        self._write_metrics(losses, data_time) # Collates loss_dict


    def get_updated_cfg_for_model_selection(self, cfg, dataset_name, seed_offset):
        # cfg = setup(args)
        cfg.defrost()
        cfg.SEED = cfg.SEED + seed_offset  # update so new random selection
        cfg.DATASETS.TRAIN = (dataset_name,)
        cfg.INPUT.MIN_SIZE_TRAIN = 1024
        cfg.INPUT.MAX_SIZE_TRAIN = 1024
        cfg.INPUT.RANDOM_FLIP = "none"
        cfg.DATASETS.TEST = cfg.DATASETS.TRAIN
        cfg.freeze()
        return cfg


    def update_cfg_loss(self, cfg):
        cfg.defrost()
        vanilla_cfg = get_cfg()
        #add_aldi_config(vanilla_cfg)
        #cfg.AUG = vanilla_cfg.AUG
        #cfg.DOMAIN_ADAPT = vanilla_cfg.DOMAIN_ADAPT
        #cfg.EMA = vanilla_cfg.EMA
        cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "giou"
        cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "giou"
        cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = True
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
    
    
def save_pseudo_coco_file(model_weights, results_path, dataset_name, score_threshold=0.5):
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


def calc_box_loss(dataset_name, perturbed_results_dir, threshold=0.5):
    # compare results and calc box loss
    logger.info("model_selection: Calculating box loss")
    dataset_images = DatasetCatalog.get(dataset_name)
    dataset_metadata = MetadataCatalog.get(dataset_name)
    # unmap the category mapping ids for COCO
    if hasattr(dataset_metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in dataset_metadata.thing_dataset_id_to_contiguous_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa    
        
    # Remove annotations which are the ground truth
    #for d in dataset_images:
    #    d["id"] = d["image_id"]
    #    del d["image_id"]
    #    del d['annotations']
        
    # Get annotations from results from last test run  
    coco_results_file = os.path.join(perturbed_results_dir, "coco_instances_results.json")
    with open(coco_results_file, 'r') as file:
        logger.info(f"model_selection: Calculating box loss - Loading results from {coco_results_file}")
        perturbed_annotations = json.load(file)
    perturbed_ann_by_image = defaultdict(list)
    for ann in perturbed_annotations:
        if ann['score'] > threshold:
            ann['category_id'] = dataset_metadata.thing_dataset_id_to_contiguous_id[ann['category_id']]
            perturbed_ann_by_image[ann['image_id']].append(ann)
    
    #match_quality = 0
    giou_losses, smooth_l1_losses, iou_losses = [], [], []
    no_matches = []
    for instance in dataset_images:
        perturbed_anns = perturbed_ann_by_image[instance["image_id"]]
        perturbed_boxes = np.array([BoxMode.convert(a["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for a in perturbed_anns])
        pseudo_anns = instance['annotations']
        pseudo_boxes = np.array([BoxMode.convert(a["bbox"], a["bbox_mode"], BoxMode.XYXY_ABS) for a in pseudo_anns])
        #max_match = min(len(perturbed_anns), len(pseudo_anns))
        # Match boxes using Faster-RCNN matching quality (from detectron2.modelling.proposal_generator.rpn)
        match_quality_matrix = pairwise_iou(Boxes(perturbed_boxes), Boxes(pseudo_boxes)) # gt rows, perturb cols
        match_row, match_col = linear_sum_assignment(match_quality_matrix, maximize=True)
        if len(match_row) == 0:
            no_matches.append(instance["image_id"])
            continue
        #match_quality += match_quality_matrix[match_row, match_col].sum().numpy().tolist() / max_match
        giou_l = giou_loss(torch.Tensor(perturbed_boxes[match_row,:]), torch.Tensor(pseudo_boxes[match_col, :]), reduction="mean").item()
        iou_l = complete_box_iou_loss(torch.Tensor(perturbed_boxes[match_row,:]), torch.Tensor(pseudo_boxes[match_col, :]), reduction="mean").item()
        smooth_l1_l = F.smooth_l1_loss(torch.Tensor(perturbed_boxes[match_row,:]), torch.Tensor(pseudo_boxes[match_col, :]), reduction="mean").item()
        giou_losses.append(giou_l)
        smooth_l1_losses.append(smooth_l1_l)
        iou_losses.append(iou_l)
    logger.info(f"model_selection: No matched boxes found for image_ids {no_matches}")
    return np.mean(giou_losses), np.mean(smooth_l1_losses), np.mean(iou_losses)


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
        new_ds_indices = indices[i * sample_size:(i + 1) * sample_size]
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





