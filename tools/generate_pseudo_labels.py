#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Copied directly from detectron2/tools/train_net.py except where noted.
"""
import glob
import pprint
from datetime import timedelta
import os
import copy
import functools
import logging
import json
from collections import defaultdict, OrderedDict

import numpy as np
from detectron2.engine import LRScheduler
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data.build import DatasetCatalog, MetadataCatalog
from aldi.methodsDirectory2Fast import perturb_model_parameters

import neptune
from neptune_detectron2 import NeptuneHook
try:
    from neptune.utils import stringify_unsupported
except ImportError:
    from neptune.new.utils import stringify_unsupported
    
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.events import EventStorage, get_event_storage
from aldi.config import add_aldi_config
from aldi.trainer import ALDITrainer

### Need to keep these especially datasets
import aldi.datasets # register datasets with Detectron2
import aldi.model # register ALDI R-CNN model with Detectron2
import aldi.backbone # register ViT FPN backbone with Detectron2


def setup(args):
    """
    Copied from detectron2/tools/train_net.py
    """
    cfg = get_cfg()

    ## Change here
    add_aldi_config(cfg)
    ## End change

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args, score_threshold=0.5, n_perturbation=3):
    """
    Copied from detectron2/tools/train_net.py
    But replace Trainer with DATrainer and disable TTA.
    """
    cfg = setup(args)
    #TEST: ("squidle_urchin_2011_test",)
    #UNLABELED: ("squidle_urchin_2009_train","squidle_east_tas_urchins_train",)
    dataset_name = cfg.DATASETS.TEST[0]
    trainer_cls = ALDITrainer
    
    # Get list of model weights - based on matching path
    pattern = os.path.join(cfg.OUTPUT_DIR, 'model*.pth')
    model_paths = glob.glob(pattern)
    print(model_paths)

    outputs = OrderedDict()
    for model_weights in sorted(model_paths): 
        # Set the dataset to Test to create pseudo label coco file
        cfg.defrost()
        cfg.DATASETS.TEST = (dataset_name,)
        cfg.MODEL.WEIGHTS = model_weights
        cfg.freeze()
        
        og_test_dataset = DatasetCatalog.get(dataset_name)
    
        # Create trainer and run test on dataset for eval with gt
        trainer = trainer_cls(cfg)
        trainer.resume_or_load(resume=args.resume)
        results = trainer_cls.test(cfg, trainer.model)
        outputs[model_weights] = results['bbox'] # Assumes bbox results
        del trainer
    
        dataset_file, model_dataset_name = save_pseudo_coco_file(cfg, dataset_name, score_threshold)
        register_coco_instances(model_dataset_name, {}, dataset_file, "./")
    
        #losses_base = train_one_step(trainer)
        
        losses_perturbed = defaultdict(list)
        update_cfg_for_one_step(cfg, model_dataset_name)
        for _ in range(n_perturbation):
            trainer = build_trainer_with_perturbation(cfg, trainer_cls)
            losses = train_one_step(trainer)
            for k in losses.keys():
                vals = losses_perturbed[k]
                vals.append(losses[k][0])
            del trainer
        losses_perturbed = {k: (float(np.average(v)), float(np.std(v))) for k, v in losses_perturbed.items() if "loss" in k}
    
        #loss_diff = loss_difference(losses_base, losses_perturbed)
        print(f"Perturbed loss: {losses_perturbed}")
        outputs[model_weights].update(losses_perturbed)
        
    pprint.pprint(outputs)
    
    output_by_eval = {}
    model_weights = list(outputs.keys())
    all = [outputs[mw]['AP50'] for mw in model_weights]
    max_idx = all.index(max(all))
    output_by_eval['AP50_best'] = outputs[model_weights[max_idx]]['AP50']
    min_idx = all.index(min(all))
    output_by_eval['AP50_worst'] = outputs[model_weights[min_idx]]['AP50']    
    
    loss_names = [l for l in outputs[model_weights[0]].keys() if "loss" in l]
    for loss in loss_names:
        all = [outputs[mw][loss][0] for mw in model_weights]
        min_idx = all.index(min(all))
        output_by_eval[loss] = outputs[model_weights[min_idx]]['AP50']
    pprint.pprint(output_by_eval)
    

def loss_difference(base_losses, new_losses, absolute=True):
    loss_difference = {}
    for key in base_losses.keys():
        if key in new_losses and "loss" in key:
            loss_difference[key] = new_losses[key][0] - base_losses[key][0]
            if absolute:
                loss_difference[key] = abs(loss_difference[key])
    return loss_difference
    

def train_one_step(trainer):
    # Run training step to calculate all losses
    trainer._hooks = [h for h in trainer._hooks if isinstance(h, LRScheduler)]
    with EventStorage(10) as trainer.storage:
        try:
            trainer.before_train()
            trainer.before_step()
            trainer.run_step()
            trainer.after_step()
        except Exception as ex:
            print(f"Exception during training: {ex}")
            raise
    return trainer.storage.latest()


def update_cfg_for_one_step(cfg, model_dataset_name):
    cfg.defrost()
    cfg.DATASETS.TEST = (model_dataset_name,)
    cfg.DATASETS.TRAIN = (model_dataset_name,)
    cfg.DATASETS.BATCH_CONTENTS = ("labeled_weak",) # Will apply weak augmentation
    cfg.SOLVER.MAX_ITER = 1
    cfg.TEST.EVAL_PERIOD = 1
    cfg.EMA.ENABLED = False
    cfg.freeze()


def build_trainer_with_perturbation(cfg, trainer_cls):
    # Build trainer with a perturbed model.  Reruns the steps 
    # to build the trainer after model perturbation 
    # to make sure all the hooks / optimizer are correct.
    
    trainer = trainer_cls(cfg)
    trainer.resume_or_load(resume=False)
    
    # Perturb model
    model_copied = copy.deepcopy(trainer.model)
    model_copied = perturb_model_parameters(model_copied)
    trainer.model = model_copied
    optimizer = trainer.build_optimizer(cfg, model_copied)
    
    ### Change is here ###
    model = trainer.create_ddp_model(model_copied, broadcast_buffers=False, cfg=cfg,
                                     find_unused_parameters=cfg.MODEL.FIND_UNUSED_PARAMETERS)
    ###   End change   ###
    
    ## Change is here ##
    trainer._trainer = trainer._create_trainer(cfg, model, trainer.data_loader, optimizer)
    ##   End change   ##
    trainer.scheduler = trainer.build_lr_scheduler(cfg, optimizer)
    
    ## Change is here ##
    trainer.checkpointer = trainer._create_checkpointer(model, cfg)
    ##   End change   ##
    
    trainer.start_iter = 10 # Something greater than zero
    trainer.max_iter = cfg.SOLVER.MAX_ITER
    trainer.cfg = cfg
    trainer._hooks = []
    trainer.register_hooks(trainer.build_hooks())
    return trainer


def save_pseudo_coco_file(cfg, dataset_name, score_threshold=0.5):
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
    coco_results_file = os.path.join(cfg.OUTPUT_DIR, "inference/coco_instances_results.json")
    with open(coco_results_file, 'r') as file:
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
        'model': cfg.MODEL.WEIGHTS,
        'dataset': dataset_name,
        'description': f"Pseudo labels generated from model {cfg.MODEL.WEIGHTS} on dataset {dataset_name}",
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
    model_weights = 'zz'.join(cfg.MODEL.WEIGHTS[:-4].split('/')[1:])
    model_dataset_name = f"{model_weights}__{dataset_name}"
    dataset_file = os.path.join(cfg.OUTPUT_DIR, f"psuedo_{model_dataset_name}.json")

    with open(dataset_file, "w") as fp:
        json.dump(psuedo_coco_dataset, fp)
    print(
        f"Saving new coco file with psuedo labels from dataset {dataset_name} and model weights {cfg.MODEL.WEIGHTS} at {dataset_file}")
    return dataset_file, model_dataset_name


@functools.lru_cache()
def setup_neptune_logging(project, api_token, freq, tags, group_tags, eval_only=False):
    run = neptune.init_run(project=project, api_token=api_token)
    if len(tags) > 0:
        run['sys/tags'].add(tags.split(','))
    if len(group_tags) > 0:
        run['sys/group_tags'].add(group_tags.split(','))
    hook = NeptuneHook(run=run, log_model=False, metrics_update_freq=freq)
    hook.base_handler["config/EVAL_ONLY"] = eval_only
    return run, hook



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        timeout=timedelta(minutes=10), # added for debugging
        args=(args,),
    )
