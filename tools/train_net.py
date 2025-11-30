#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Copied directly from detectron2/tools/train_net.py except where noted.
"""
from datetime import timedelta
import time
import os
import sys
import functools
import logging

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results

import neptune
from neptune_detectron2 import NeptuneHook
try:
    from neptune.utils import stringify_unsupported
except ImportError:
    from neptune.new.utils import stringify_unsupported

from aldi.checkpoint import DetectionCheckpointerWithEMA
from aldi.config import add_aldi_config
from aldi.config_fcos import add_fcos_config
from aldi.ema import EMA
from aldi.trainer import ALDITrainer
import aldi.datasets # register datasets with Detectron2
import aldi.model # register ALDI R-CNN model with Detectron2
import aldi.backbone # register ViT FPN backbone with Detectron2
from aldi.split_datasets import split_train_data, split_test_data
import aldi.datasets_benthic # register datasets with Detectron2
import aldi.distill_saod
from aldi.fcos.fcos import FCOS
import aldi.fcos.align
import aldi.fcos.distill


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def setup(args):
    """
    Copied from detectron2/tools/train_net.py
    """
    cfg = get_cfg()

    ## Change here
    
    # enclose YOLO in a try/except because we want the extra pip dependencies to be optional
    #try:
    #    from aldi.yolo.helpers import add_yolo_config
    #    import aldi.yolo.align # register align mixins with Detectron2
    #    import aldi.yolo.distill # register distillers and distill mixins with Detectron2
    #    add_yolo_config(cfg)
    #except:
    #    print("Could not load YOLO library.")
        
    try:
        import aldi.detr.align
        import aldi.detr.distill
        from aldi.detr.helpers import add_deformable_detr_config
        add_deformable_detr_config(cfg)
    except ImportError:
        print("Failed to load DETR.  Skipping...")
    add_fcos_config(cfg)    
    add_aldi_config(cfg)
    ## End change

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    split_train_data(cfg)
    split_test_data(cfg)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    """
    Copied from detectron2/tools/train_net.py
    But replace Trainer with DATrainer and disable TTA.
    """
    cfg = setup(args)

    if args.eval_only:
        model = ALDITrainer.build_model(cfg)
        ## Change here
        ckpt = DetectionCheckpointerWithEMA(model, save_dir=cfg.OUTPUT_DIR)
        if cfg.EMA.ENABLED and cfg.EMA.LOAD_FROM_EMA_ON_START:
            ema = EMA(ALDITrainer.build_model(cfg), cfg.EMA.ALPHA)
            ckpt.add_checkpointable("ema", ema)
        ckpt.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)

        # Build evaluators
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            iou_evaluator = ALDITrainer.build_iou_evaluator(cfg, dataset_name)
            iou_res = ALDITrainer.test(cfg, model, iou_evaluator)
        ## End change
        res = ALDITrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError("TTA not supported")
        if comm.is_main_process():
            verify_results(cfg, res)

            # Neptune logging results
            for k in res.keys():
                res[k].update(iou_res[k])
            run, hook = setup_neptune_logging(cfg.LOGGING.PROJECT, cfg.LOGGING.API_TOKEN, cfg.LOGGING.ITERS, cfg.LOGGING.TAGS, cfg.LOGGING.GROUP_TAGS, args.eval_only)
            hook.base_handler["config"] = stringify_unsupported(cfg)
            for j, u in res.items():
                for k, v in u.items():
                    hook.base_handler[f"metrics/{j}/{k}"].append(v)
            run.stop()
        return res

    cfg = adjust_lr_for_ims_per_gpu(cfg, comm.get_world_size())
    trainer = ALDITrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if comm.is_main_process():
        run, hook = setup_neptune_logging(cfg.LOGGING.PROJECT, cfg.LOGGING.API_TOKEN, cfg.LOGGING.ITERS, cfg.LOGGING.TAGS, cfg.LOGGING.GROUP_TAGS, cfg.LOGGING.PROXY)
        trainer.register_hooks([hook])
    out = trainer.train()
    if comm.is_main_process():
        run.stop()
    return out

    # Neptune logging
@functools.lru_cache()
def setup_neptune_logging(project, api_token, freq, tags, group_tags, proxy_server="", eval_only=False):
    import subprocess
    proxies = {}
    for proxy in ['http', 'https']:
        result = subprocess.run(f"env | grep -i {proxy}_proxy=", shell=True, capture_output=True, text=True)
        if result.stdout:
            proxy_value = result.stdout.strip().split('=', 1)[1]
            proxies[proxy] = proxy_value
        else:
            proxies[proxy] = ""
    print(f"proxies={proxies}")
    run = neptune_init_with_retry(project=project, api_token=api_token, proxies=proxies)
    if len(tags) > 0:
        run['sys/tags'].add(tags.split(','))
    if len(group_tags) > 0:
        run['sys/group_tags'].add(group_tags.split(','))
    hook = NeptuneHook(run=run, log_model=False, metrics_update_freq=freq)
    hook.base_handler["config/EVAL_ONLY"] = eval_only
    return run, hook
    
def neptune_init_with_retry(project, api_token, proxies, max_retries=100, delay=1):
    retries = 0
    while retries < max_retries:
        try:
            run = neptune.init_run(project=project, api_token=api_token, proxies=proxies)
            print("Neptune run initialized successfully.")
            return run
        except neptune.exceptions.CannotResolveHostname as e:
            retries += 1
            print(f"Attempt {retries} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise Exception("Failed to initialize Neptune run after multiple attempts.")

def adjust_lr_for_ims_per_gpu(cfg, world_size):
    """
    SOLVER.IMS_PER_GPU introduces mini-batch loops if ngpus * IMS_PER_GPU != IMS_PER_BATCH.
    This means the number of iterations does not need to be changed when ngpus * IMS_PER_GPU < IMS_PER_BATCH
    ie there are more 'mini-iterations' but the LR needs to be scaled if BACKWARD_AT_END is false, 
    as the effective global batch size is smaller.
    """
    if world_size < (cfg.SOLVER.IMS_PER_BATCH / cfg.SOLVER.IMS_PER_GPU) and not cfg.SOLVER.BACKWARD_AT_END:
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        scale = world_size * cfg.SOLVER.IMS_PER_GPU / cfg.SOLVER.IMS_PER_BATCH
        old_lr = cfg.SOLVER.BASE_LR
        lr = cfg.SOLVER.BASE_LR = old_lr * scale
        logger = logging.getLogger(__name__)
        logger.info(
            f"Worldsize ({world_size} is smaller than IMS_PER_BATCH ({cfg.SOLVER.IMS_PER_BATCH}) / IMS_PER_GPU ({cfg.SOLVER.IMS_PER_GPU})"
            f"Scaling LR from {old_lr} to {lr} "
        )
        if frozen:
            cfg.freeze()
    return cfg

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
