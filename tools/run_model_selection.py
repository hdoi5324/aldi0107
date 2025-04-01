#!/usr/bin/env python


import glob
import pprint
from datetime import timedelta
import os
import functools
import time
import logging
from collections import OrderedDict
import pandas as pd


from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
import detectron2.utils.comm as comm

import neptune
from neptune_detectron2 import NeptuneHook

try:
    from neptune.utils import stringify_unsupported
except ImportError:
    from neptune.new.utils import stringify_unsupported
    
from aldi.config import add_aldi_config
import aldi.datasets # register datasets with Detectron2
import aldi.model # register ALDI R-CNN model with Detectron2
import aldi.backbone # register ViT FPN backbone with Detectron2

from model_selection.model_selection import ModelSelection, get_dataset_samples
logger = logging.getLogger("detectron2")


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

def main(args):
    """
    Copied from detectron2/tools/train_net.py
    But replace Trainer with DATrainer and disable TTA.
    """
    cfg = setup(args)
    selector = ModelSelection(cfg, list(cfg.DATASETS.TRAIN), 
                              target_ds=list(cfg.DATASETS.TEST), 
                              n_samples=cfg.MODEL_SELECTION.SAMPLE, 
                              dropout=cfg.MODEL_SELECTION.DROPOUT, 
                              n_perturbations=cfg.MODEL_SELECTION.N_PERTURBATIONS,
                              )
    
    # Get list of model weights - based on matching path
    model_paths = sorted(glob.glob(os.path.join(cfg.OUTPUT_DIR, 'model_0*999.pth')))
    logger.info(f"model_selection: Models found: {model_paths}")

    # Log to Neptune
    if comm.is_main_process() and False:
        run, hook = setup_neptune_logging("ACFRmarine/Unsupervised-Model-Selection", cfg.LOGGING.API_TOKEN, cfg.LOGGING.ITERS, cfg.LOGGING.TAGS, cfg.LOGGING.GROUP_TAGS, cfg.LOGGING.PROXY)
        hook.base_handler["config"] = stringify_unsupported(cfg) 
        # todo: record ModelSelection parameters OR put them in config.
    else:
        run = None
        
    outputs = OrderedDict()
    for m_idx, model_weights in enumerate(model_paths):
        model_output_src = selector.run_model_selection(model_weights, source=True, neptune_run=run) 
        outputs[model_weights] = model_output_src
        model_output_tgt = selector.run_model_selection(model_weights, source=False, neptune_run=run) 
        outputs[model_weights].update(model_output_tgt)
        
    if comm.is_main_process():
        logger.info(f"model_selection: output: {pprint.pformat(outputs)}")
        
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
        output_file = os.path.join(cfg.OUTPUT_DIR, 'model_selection/model_selection.json')
        df.to_json(output_file)
        logger.info(f"model_selection: Saved results to {output_file}")

        if run is not None:    
            run['metrics/tgt_datasets'] = selector.sampled_tgt
            run['metrics/src_datasets'] = selector.sampled_src
            run['metrics/model_weights'] = model_paths
            run.stop()



@functools.lru_cache()
def setup_neptune_logging(project, api_token, freq, tags, group_tags, proxy_server="", eval_only=False):
    proxies = {}
    if len(proxy_server) > 0:
        proxies = { 
                      "http"  : proxy_server, 
                      "https" : proxy_server, 
                    }
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


def log_nested_dict(run, dictionary, parent_key=""):
# Recursive function to log nested dictionary
    for key, value in dictionary.items():
        full_key = f"{parent_key}/{key}" if parent_key else key
        if isinstance(value, dict):
            log_nested_dict(run, value, full_key)
        else:
            run[full_key].append(value)



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
