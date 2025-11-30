import sys
import datetime
import json
import logging
import os
import pprint
import time
import copy
import numpy as np
from collections import defaultdict
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from fvcore.common.checkpoint import _IncompatibleKeys
from scipy.optimize import linear_sum_assignment

import torch.nn.functional as F
from detectron2.modeling import GeneralizedRCNN
from detectron2.data.build import filter_images_with_only_crowd_annotations, build_detection_test_loader
from detectron2.data.samplers import InferenceSampler
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.modeling import build_model
from detectron2.utils.logger import log_every_n_seconds, _log_api_usage
from fvcore.nn.giou_loss import giou_loss

from detectron2.utils.events import EventStorage
from detectron2.structures import pairwise_iou, Boxes, BoxMode
from detectron2.evaluation.testing import flatten_results_dict

from aldi.config import add_aldi_config
from aldi.config_aldi_only import add_aldi_only_config

#from aldi.config import add_aldi_config
#from aldi.methodsDirectory2Fast import perturb_by_dropout, dropout_masks

from model_selection.ums import UMS
from model_selection.utils import build_evaluator, perturb_by_dropout, dropout_masks, perturb_model_parameters
from aldi.split_datasets import register_coco_instances_with_split

# Override box_loss methods to use mean
from .box_loss import _mean_dense_box_regression_loss, classifier_loss_on_gt_boxes, get_outputs_with_image_id
current_module = sys.modules['detectron2.modeling.proposal_generator.rpn']
setattr(current_module, '_dense_box_regression_loss', _mean_dense_box_regression_loss)
current_module = sys.modules['detectron2.modeling.roi_heads.fast_rcnn']
setattr(current_module, '_dense_box_regression_loss', _mean_dense_box_regression_loss)

from model_selection.fast_rcnn import fast_rcnn_inference_single_image_all_scores

DEBUG = False
debug_dict = {}
logger = logging.getLogger("detectron2")


class ModelSelection:
    def __init__(self, cfg, source_ds=[], target_ds=None):
        self.storage: EventStorage
        
        _log_api_usage("trainer." + self.__class__.__name__)    
        
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.cfg = self.update_cfg_loss(cfg)
        self.model = build_model(self.cfg)
        self.model.eval()
        self.perturb_method = perturb_by_dropout if cfg.UMS.PERTURB_TYPE == "dropout" else perturb_model_parameters

        if cfg.UMS.N_TRANSFORMED_SOURCE > 0:
            new_source_ds = []
            for i in range(cfg.UMS.N_TRANSFORMED_SOURCE):
                new_source_ds += [f"{s_ds}_transformed{i}" for s_ds in source_ds]
            new_source_ds += source_ds
            source_ds = new_source_ds
        self.sampled_src = get_dataset_samples(source_ds, n=cfg.UMS.N_SAMPLE) if source_ds is not None else []
        #self.sampled_src = self.sampled_src[:min(len(self.sampled_src), 6)] # Debug line
        self.sampled_tgt = get_dataset_samples(target_ds, n=1000000) if target_ds is not None else [] # Use all target data
        self.evaluation_dir = os.path.join(cfg.OUTPUT_DIR, "model_selection")
  
        
    def run_ums(self, model_weights, source=True, neptune_run=None):
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
            evaluator = build_evaluator(cfg, dataset_name, self.evaluation_dir, do_eval=True)
            dataset = DatasetCatalog.get(dataset_name)
            data_loader = build_detection_test_loader(
                dataset=dataset,
                mapper=DatasetMapper(cfg, True),
                sampler=InferenceSampler(len(dataset)),
                num_workers=cfg.DATALOADER.NUM_WORKERS)
            ums_calculator = UMS(cfg, self.model, data_loader, evaluator, perturbation_types=['das', 'dropout'])
            ums_calcs = ums_calculator.calculate_measures()
            
            if neptune_run is not None:
                flattened_results = flatten_results_dict(ums_calcs)
                for k, v in flattened_results.items():
                    neptune_run[f"metrics/{model_shortname}/{unique_dataset_name}/{k}"] = v
                    
            logger.info(f"model_selection: UMS for {dataset_name}, {model_shortname}: {pprint.pformat(ums_calcs)}")
            outputs[unique_dataset_name].update(ums_calcs)
                
            iters_after_start = ds_idx + 1
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if ds_idx % info_freq == 0 or ds_idx == total - 1:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - ds_idx - 1)))
                logger.info(f"Model selection for {str(model_shortname)} done {ds_idx + 1}/{total} sampled datasets. Total : {total_seconds_per_iter:.4f} s/iter. ETA={eta}")
        return outputs


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
        #cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "giou"
        #cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "giou"
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


def get_model_shortname(model_weights):
    return os.path.basename(model_weights)[:-4]


def load_model_weights(path, model):
    checkpointer = DetectionCheckpointer(model)  
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
