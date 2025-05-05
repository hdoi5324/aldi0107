import os
import json
import pprint
import pandas as pd

import torch
from detectron2.checkpoint import DetectionCheckpointer
from fvcore.common.checkpoint import _IncompatibleKeys

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import DatasetEvaluators
from aldi.evaluation import Detectron2COCOEvaluatorAdapter
from detectron2.config import get_cfg
from aldi.config import add_aldi_config
from aldi.config_aldi_only import add_aldi_only_config
from detectron2.engine import default_setup

def setup(args):
    """
    Copied from detectron2/tools/train_net.py
    """
    cfg = get_cfg()

    ## Change here
    add_aldi_config(cfg)
    add_aldi_only_config(cfg)  # adds a couple of keys as configs have diverged.
    ## End change

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

        
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
        mask = (torch.rand(weights.shape[0], weights.shape[1], 1, 1) > p).float().to(weights.device)
        mask = mask.expand_as(weights)
        mask = mask / (1 - p)
    else:
        mask = torch.ones((weights.shape[0], weights.shape[1], 1, 1)).float().to(weights.device)
        mask = mask.expand_as(weights)
    return mask


def dropout_masks(module, p=.1, weights_filter='res4.2.conv3.weight'):
    state_dict = module.state_dict()
    last_layer_parameters = [(k, v) for k, v in state_dict.items() if weights_filter in k]
    mask_dict = {}
    for (k, w) in last_layer_parameters:
        mask_dict[k] = dropout_mask_along_channel(w, p)
    return mask_dict


def perturb_by_dropout(module, p=.1, mask_dict={}, weights_filter='res4.2.conv3.weight'):
    state_dict = module.state_dict()
    last_layer_parameters = [(k, v) for k, v in state_dict.items() if weights_filter in k]
    for (k, w) in last_layer_parameters:
        state_dict[k] = w * mask_dict[k] if k in mask_dict else w * dropout_mask_along_channel(w, p)
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