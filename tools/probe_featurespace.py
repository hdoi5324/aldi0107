#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Copied directly from detectron2/tools/train_net.py except where noted.
"""
import glob
import os
import pprint
from collections import OrderedDict
from datetime import timedelta
from typing import OrderedDict
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import inference_on_dataset
from detectron2.data.build import get_detection_dataset_dicts, build_detection_test_loader

from aldi.checkpoint import DetectionCheckpointerWithEMA
from aldi.config import add_aldi_config
from aldi.dropin import DatasetMapper
from aldi.ema import EMA
from aldi.trainer import ALDITrainer
from aldi.methodsDirectory2Fast import FIS, PDR, pooled_features, pred_roi_pooled_features, \
    fast_rcnn_inference_single_image
import aldi.datasets # register datasets with Detectron2
import aldi.model # register ALDI R-CNN model with Detectron2
import aldi.backbone # register ViT FPN backbone with Detectron2



try:
    from sklearn.decomposition import PCA
except ModuleNotFoundError:
    print("""
          Feature visualization requires scikit-learn to be installed.
          Please insteall scikit-learn (e.g. run `pip install scikit-learn`)
          and try again.
    """)


def ice_score_rpn_single(x, net, lam=0.55):
    # Random batch index .
    _, features, proposals, _ = x
    inputs_a = pooled_features(features, proposals, net)

    # Obtain model predictions and hard pseudo labels .
    boxes, scores, image_shape = pred_roi_pooled_features(inputs_a, proposals,
                                                          net)  # network output for multiple images
    # todo: Maybe update for batchsize > 1.  Currently assumes single image.
    keep_idx = fast_rcnn_inference_single_image(boxes[0], scores[0], image_shape[0])  # Need boxes for NMS

    if len(keep_idx) > 0:
        inputs_a = inputs_a[keep_idx]
        pl_a = scores[0][keep_idx].max(dim=1)[1]  # network class

        rand_idx = torch.randperm(inputs_a.shape[0])  # random.choice(keep_idx) 
        inputs_b = inputs_a[rand_idx]  # Used in mixup        
        pl_b = pl_a[rand_idx]  # .reshape([1,])# network class for b sample

        # Intra - cluster mixup .
        same_idx = (pl_a == pl_b).nonzero()
        # Inter - cluster mixup .
        diff_idx = (pl_a != pl_b).nonzero()
        # Mixup with images and hard pseudo labels .
        mix_inputs = lam * inputs_a + (1 - lam) * inputs_b  # combine input maps
        if lam > 0.5:
            mix_labels = pl_a  # if more a is used, mixed label is a 
        else:
            mix_labels = pl_b  # torch.hstack([pl_b] * inputs_a.shape[0]) # if more b is used, mixed label is b
        # Obtain predictions for the mixed samples .
        _, mix_pred, _ = pred_roi_pooled_features(mix_inputs, [proposals[0][keep_idx]],
                                                  net)  # todo: update to deal with batchsize > 1
        mix_pred_labels = mix_pred[0].max(dim=1)[1]
        # Calculate ICE scores for two - dimensional probing .
        ice_same = torch.sum(mix_pred_labels[same_idx] == mix_labels[same_idx]) / same_idx.shape[0] if same_idx.shape[
                                                                                                           0] > 0 else torch.tensor(
            0.)
        ice_diff = torch.sum(mix_pred_labels[diff_idx] == mix_labels[diff_idx]) / diff_idx.shape[0] if diff_idx.shape[
                                                                                                           0] > 0 else torch.tensor(
            0.)
        # if same_idx.shape[0] ==0 or diff_idx.shape[0] == 0:
        #    print(f"All one type.  No same or no different classes.")
        return ice_same.detach().cpu().numpy(), ice_diff.detach().cpu().numpy()
    else:
        return 0.0, 0.0


def od_ice(model, data_loader, evaluator, lam=0.55, repeat=1):
    # Use hooks to collect image and proposal features        
    pooling_method = F.avg_pool2d  # or F.max_pool2d


    same_ice, diff_ice = [], []
    for _ in range(repeat):
        proposal_level_probe = []
        image_level_features = []
        proposal_level_features = []
        
        # Get image and proposal features for PCA
        backbone_hook_handle = model.backbone.register_forward_hook(
            lambda module, input, output: image_level_features.extend(
                pooling_method(output[sorted(output.keys())[-1]], kernel_size=output[sorted(output.keys())[-1]].shape[2:4])[
                    ..., 0, 0].detach().cpu().numpy()))
        roi_heads_hook_handle = model.roi_heads.box_pooler.register_forward_hook(
            lambda module, input, output: proposal_level_features.extend(
                pooling_method(output, kernel_size=output.shape[2:4])[..., 0, 0].detach().cpu().numpy()))
        
        # Use hook to calculate ICE using Mixup of proposals
        roi_heads_probe_hook = model.roi_heads.register_forward_hook(
            lambda module, input, output: proposal_level_probe.append(ice_score_rpn_single(input, module, lam)))
        evaluation = inference_on_dataset(model, data_loader, evaluator=evaluator)
        same_diff_ice = np.average(np.array(proposal_level_probe), axis=0).tolist()
        same_ice.append(same_diff_ice[0])
        diff_ice.append(same_diff_ice[1])

        # clean up hooks
        backbone_hook_handle.remove()
        roi_heads_hook_handle.remove()
        roi_heads_probe_hook.remove()
    
    results_dict = {'same_ice': same_ice,
                    'diff_ice': diff_ice,
                    'eval': evaluation['bbox']['AP50'],
                    'evaluation': evaluation,}
    print(f"Same / Diff ICE {pprint.pprint(results_dict)}")

    return results_dict, image_level_features, proposal_level_features


# Perform model selection based on ICE scores .
def mixVal(outputs, descending=True):
    # Calculate ICE scores for all candidate models .
    avg_ice_rank = []
    models = list(outputs.keys())
    for m in models:
        avg_ice_rank.append((outputs[m]['same_ice_rank'] + outputs[m]['diff_ice_rank'])/2)
    ice_rank = torch.argsort(torch.argsort(torch.tensor(avg_ice_rank), descending=descending))  
    for i, m in enumerate(models):
        outputs[m]['avg_ice_rank_raw'] = avg_ice_rank[i] 
        outputs[m]['avg_ice_rank'] = ice_rank[i].item()
    return outputs


def normalize(numbers):
    min_num = min(numbers)
    max_num = max(numbers)
    return torch.tensor([(x - min_num) / (max_num - min_num) for x in numbers])


def das(outputs, lam=1.0, index=1):
    models = list(outputs.keys())
    pdrs = torch.tensor([1 for model_name in models])
    #pdrs = torch.tensor([outputs[model_name]['pdr']['score'] for model_name in models])
    #pdrs = normalize(pdrs)
    fiss = -1 * torch.tensor(
        [np.average(outputs[model_name]['giou_cost'][index]) for model_name in models])
    fiss = normalize(fiss)
    dass = fiss + lam * pdrs
    pdr_rank = torch.argsort(torch.argsort(pdrs, descending=True))
    fis_rank = torch.argsort(torch.argsort(fiss, descending=True))
    das_rank = torch.argsort(torch.argsort(dass, descending=True))
    for i, model_name in enumerate(models):
        outputs[model_name]['pdr_rank'] = pdr_rank[i].item()
        outputs[model_name]['fis_rank'] = fis_rank[i].item()
        outputs[model_name]['das_rank'] = das_rank[i].item()
    return outputs


def add_rank(outputs, key_to_rank, descending=True):
    models = list(outputs.keys())
    avg_model_values = []
    for m in models:
        if isinstance(outputs[m][key_to_rank], list):
            avg_model_values.append(np.average(outputs[m][key_to_rank]))
        else:
            avg_model_values.append(outputs[m][key_to_rank])
    rank = torch.argsort(torch.argsort(torch.tensor(avg_model_values), descending=descending))
    for i, m in enumerate(models):
        outputs[m][f"{key_to_rank}_rank"] = rank[i].item()
    return outputs

def get_model_df(outputs):
    models = list(outputs.keys())
    df_keys = [k for k in sorted(list(next(iter(outputs.values())))) if k not in ['evaluation']]
    df_rows = []
    for m in models:
        model_row = []
        for k in df_keys:
            if isinstance(outputs[m][k], list):
                model_row.append(np.average(outputs[m][k]))
            elif isinstance(outputs[m][k], torch.Tensor):
                model_row.append(outputs[m][k].item())
            else:
                model_row.append(outputs[m][k])
        df_rows.append(model_row)
    
    df = pd.DataFrame(data=df_rows, columns=df_keys)
    df.index = models
    return df


def summarise_outputs(outputs, last_checkpoint="model_final"):
    models = list(outputs.keys())
    last_model = [m for m in models if last_checkpoint in m][0]
    last_model_results = outputs[last_model]
    last_model_eval = last_model_results['AP50']
    results = []
    df_heading = []

    def summary_data(out_dict, model, measure=None, last_eval=None):
        model_output = out_dict[model]
        eval_measure = model_output['AP50']
        last_eval = last_eval if last_eval is not None else eval_measure
        return_list = [eval_measure, eval_measure - last_eval, model_output["eval_rank"]]
        if measure is not None:
            return_list.append(model_output[measure])
            return_list.append(model_output[f"{measure}_rank"])
        else:
            return_list.append(model_output["eval"])
            return_list.append(model_output[f"eval_rank"])
        return return_list

    # Last checkpoint
    best_eval_model = models[np.argmax([outputs[m]['AP50'] for m in models])]
    worst_eval_model = models[np.argmin([outputs[m]['AP50'] for m in models])]
    
    results.append(summary_data(outputs, best_eval_model, measure="AP50"))
    results.append(summary_data(outputs, worst_eval_model, measure="AP50"))
    
    for measure in ['iou_cost']:
        best_measure_model = models[np.argmin([outputs[m][f"{measure}_rank"] for m in models])]
        results.append(summary_data(outputs, best_measure_model, last_model_eval))
    #best_pdr_model = models[np.argmin([outputs[m][d]['pdr_rank'] for m in models])]
    #best_fis_model = models[np.argmin([outputs[m][d]['fis_rank'] for m in models])]

    #best_ice_model = models[np.argmin([outputs[m][d]['ice_rank'] for m in models])]
    #best_eval_result = summary_data(outputs, best_eval_model, d, last_model_eval)
    #worst_eval_result = summary_data(outputs, worst_eval_model, d, last_model_eval)
    #best_ice_result = summary_data(outputs, best_ice_model, d, last_model_eval)
    #best_das_result = summary_data(outputs, best_das_model, d, last_model_eval)
    #best_pdr_result = summary_data(outputs, best_pdr_model, d, last_model_eval)
    #best_fis_result = summary_data(outputs, best_fis_model, d, last_model_eval)
    #results.append(np.array(
    #    [last_result, best_eval_result, worst_eval_result, best_ice_result, best_das_result, best_pdr_result,
    #     best_fis_result]))
    df_heading  [f"_AP50", f"_diff_last", f"_eval_rank", f"measure", f"measure_rank"]

    df = pd.DataFrame(data=np.hstack(results), columns=df_heading)
    df.index = ["Last", "Best", "Worst", "Measure", "Rank"]
    return df


def pca(image_level_features, proposal_level_features, cfg, model_weights):
    # transform and visualize image-level features
    pca = PCA(n_components=2).fit(np.array(image_level_features))
    image_level_features = pca.transform(image_level_features)
    plt.scatter(x=image_level_features[:, 0], y=image_level_features[:, 1],
                alpha=0.5, s=1)
    leg = plt.legend(markerscale=4)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    plt.title(f"Image-level PCA (exp. var. {pca.explained_variance_ratio_.sum():.2f})")
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, f"feature_vis_image_pca_{model_weights}.png"), bbox_inches="tight")
    plt.close("all")

    # transform and visualize proposal-level features
    pca = PCA(n_components=2).fit(np.array(proposal_level_features))
    proposal_level_features = pca.transform(proposal_level_features) 
    plt.scatter(x=proposal_level_features[:, 0], y=proposal_level_features[:, 1],
                alpha=0.1, s=1)
    leg = plt.legend(markerscale=4)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    plt.title(f"Proposal-level PCA (exp. var. {pca.explained_variance_ratio_.sum():.2f})")
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, f"feature_vis_proposal_pca_{model_weights}.png"), bbox_inches="tight")
    plt.close("all")


def setup(args):
    """
    Copied directly from detectron2/tools/train_net.py
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


def main(args, n=10, lam=0.55, repeat=3):
    """
    Runs evaluation and visualizes features.
    """
    # todo: add command line args - list of models, sample size
    cfg = setup(args)
    trainer = ALDITrainer

    # load model
    model = trainer.build_model(cfg)
    ckpt = DetectionCheckpointerWithEMA(model, save_dir=cfg.OUTPUT_DIR)
    if cfg.EMA.ENABLED and cfg.EMA.LOAD_FROM_EMA_ON_START:
        ema = EMA(trainer.build_model(cfg), cfg.EMA.ALPHA)
        ckpt.add_checkpointable("ema", ema)
    ckpt.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)

    # load datasets
    test_dataset = cfg.DATASETS.TEST[0]
    # data_loader = ALDITrainer.build_test_loader(cfg, dataset)
    dataset = get_detection_dataset_dicts(test_dataset, filter_empty=False)
    dataset = dataset[:min(n, len(dataset))]
    data_loader = build_detection_test_loader(dataset, mapper=DatasetMapper(cfg, is_train=False))

    source_dataset = get_detection_dataset_dicts(cfg.DATASETS.TRAIN,
                                                 filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS)
    source_dataset = source_dataset[:min(n, len(source_dataset))]
    source_dataloader = build_detection_test_loader(source_dataset, mapper=DatasetMapper(cfg, is_train=False))

    # Get list of model weights - based on matching path
    pattern = os.path.join(cfg.OUTPUT_DIR, 'model*.pth')
    model_paths = glob.glob(pattern)
    print(model_paths)

    outputs = OrderedDict()
    for model_weights in sorted(model_paths):
        model_name = '_'.join(model_weights.split('/')[1:])
        outputs[model_name] = {}
        model_outputs = outputs[model_name]

        ckpt.resume_or_load(model_weights, resume=args.resume)
        
        # Source eval todo: change this to source test
        source_eval = []
        for idx, dataset_name in enumerate([d.replace('train', 'test') for d in cfg.DATASETS.TRAIN]):
            ds = get_detection_dataset_dicts(dataset_name, filter_empty=False)
            ds = ds[:min(n, len(ds))]
            train_data_loader = build_detection_test_loader(ds, mapper=DatasetMapper(cfg, is_train=False))
            train_evaluator = trainer.build_evaluator(cfg, dataset_name)
            evaluation = inference_on_dataset(model, train_data_loader, train_evaluator)
            source_eval.append(evaluation['bbox']['AP50'])
        outputs[model_name]['source_eval'] = source_eval
        del train_data_loader, train_evaluator, evaluation, source_eval

        # Test evaluator
        evaluator = trainer.build_evaluator(cfg, test_dataset)
        
        # FIS model selection - 'iou_cost', 'giou_cost', 'match_quality', '
        model_outputs.update(FIS(model, data_loader, evaluator, max_repeat=repeat))

        # PDR model selection - 'pdr'
        model_outputs.update(PDR(model, data_loader, source_dataloader))

        # ICE model selection
        ckpt.resume_or_load(model_weights, resume=args.resume)
        ice_results, image_level_feat, prop_level_feat = od_ice(model, data_loader,
                                                                    evaluator, 
                                                                    lam, repeat=repeat)

        image_level_features = image_level_feat
        proposal_level_features = prop_level_feat
        model_outputs.update(ice_results)

        pca(image_level_features, proposal_level_features, cfg, model_name)
    measure_keys = list(next(iter(outputs.values())))
    for k in measure_keys:
        if k != 'evaluation':
            add_rank(outputs, k)

    outputs = mixVal(outputs) # Add combined ice ranking
    measure_keys = list(next(iter(outputs.values())))
    pprint.pprint(outputs)
    
    model_df = get_model_df(outputs)
    print(model_df.to_string(index=True))
    
    #df = summarise_outputs(outputs, "model_final.pth")
    #print(df.to_string(index=True))

    # Save outputs dict
    file_name = "probe_data"
    with open(os.path.join(cfg.OUTPUT_DIR, f'{file_name}.txt'), 'w') as file:
        pprint.pprint(outputs, stream=file)
        file.writelines(model_df.to_string(index=True))
    
    with open(os.path.join(cfg.OUTPUT_DIR, f'{file_name}.json'), 'w') as file:
        json.dump(outputs, file)

    model_df_filename = os.path.join(cfg.OUTPUT_DIR, f'{file_name}_model_df.json')
    print(f"Saving model df to {model_df_filename}")
    model_df.to_json(model_df_filename)
    model_df = pd.read_json(model_df_filename)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        timeout=timedelta(minutes=1),  # added for debugging
        args=(args,),
    )
