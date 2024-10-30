#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.


import neptune
from neptune_detectron2 import NeptuneHook
try:
    from neptune.utils import stringify_unsupported
except ImportError:
    from neptune.new.utils import stringify_unsupported

import argparse
import os
from itertools import chain
import cv2
import tqdm
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import default_setup

from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode, _create_text_labels

from aldi.checkpoint import DetectionCheckpointerWithEMA
from aldi.config import add_aldi_config
from aldi.ema import EMA
from aldi.trainer import ALDITrainer
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
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument(
        "--source",
        choices=["annotation", "dataloader"],
        required=True,
        help="visualize the annotations or the data loader (with pre-processing)",
    )
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


def main(args) -> None:
    global img
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    # Load model
    model = ALDITrainer.build_model(cfg)
    model.eval()

    ## Change here
    ckpt = DetectionCheckpointerWithEMA(model, save_dir=cfg.OUTPUT_DIR)
    if cfg.EMA.ENABLED and cfg.EMA.LOAD_FROM_EMA_ON_START:
        ema = EMA(ALDITrainer.build_model(cfg), cfg.EMA.ALPHA)
        ckpt.add_checkpointable("ema", ema)
    ckpt.resume_or_load(cfg.MODEL.WEIGHTS)

    def output(vis, fname):
        if vis is not None:
            if args.show:
                print(fname)
                cv2.imshow("window", vis.get_image()[:, :, ::-1])
                cv2.waitKey()
            else:
                filepath = os.path.join(dirname, fname)
                print("Saving to {} ...".format(filepath))
                vis.save(filepath)

    scale = 1.0
    if args.source == "dataloader":
        test_data_loader = ALDITrainer.build_test_loader(cfg, cfg['DATASETS']['TEST'])
        coco_data = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TEST]))
        coco_data = {c['image_id']: c for c in coco_data}
        for idx, batch in enumerate(test_data_loader):
            for i, inputs in enumerate(batch):
                img = cv2.imread(inputs['file_name'])
                #img = img.permute(1, 2, 0).cpu().detach().numpy()
                img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)

                visualizer = Visualizer(img, metadata=metadata, scale=scale)
                dic = coco_data.get(inputs['image_id'], None)
                if dic is not None:
                    vis = draw_dataset_dict(visualizer, dic, color=(0, 255, 0))

                # Get predictions
                # todo: set threshold
                #todo: check annotations 
                outputs = model([inputs])[0]
                target_fields = outputs["instances"].get_fields()
                pred_boxes = target_fields.get("pred_boxes", None)
                if pred_boxes is not None:
                    pred_boxes = pred_boxes.to('cpu')
                    labels = [f"{s.to('cpu'):.2f}" for s in target_fields.get("scores")]

                #labels = [metadata.thing_classes[i] for i in target_fields["pred_classes"]]
                vis = visualizer.overlay_instances(
                    labels=labels,
                    boxes=pred_boxes,
                    assigned_colors=[(0,0,1.0)]*len(labels),
                )
                output(vis, f"{os.path.basename(inputs['file_name'])}")
    else:
        dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TEST]))
        if cfg.MODEL.KEYPOINT_ON:
            dicts = filter_images_with_few_keypoints(dicts, 1)
        for dic in tqdm.tqdm(dicts):
            img = utils.read_image(dic["file_name"], "RGB")
            visualizer = Visualizer(img, metadata=metadata, scale=scale)
            vis = draw_dataset_dict(visualizer, dic)
            output(vis, os.path.basename(dic["file_name"]))




def draw_dataset_dict(visualiser, dic, color=(0, 0, 255)):
    """
    Draw annotations/segmentations in Detectron2 Dataset format.

    Args:
        dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

    Returns:
        output (VisImage): image object with visualizations.
    """
    annos = dic.get("annotations", None)
    if annos:
        if "segmentation" in annos[0]:
            masks = [x["segmentation"] for x in annos]
        else:
            masks = None
        if "keypoints" in annos[0]:
            keypts = [x["keypoints"] for x in annos]
            keypts = np.array(keypts).reshape(len(annos), -1, 3)
        else:
            keypts = None

        boxes = [
            (
                BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                if len(x["bbox"]) == 4
                else x["bbox"]
            )
            for x in annos
        ]

        category_ids = [x["category_id"] for x in annos]
        colors = [
            [x / 255 for x in color]
            for c in category_ids
        ]
        names = visualiser.metadata.get("thing_classes", None)
        labels = _create_text_labels(
            category_ids,
            scores=None,
            class_names=names,
            is_crowd=[x.get("iscrowd", 0) for x in annos],
        )
        visualiser.overlay_instances(
            labels=labels, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=colors
        )

        return visualiser.output

if __name__ == "__main__":
    args = parse_args()
    print("Command Line Args:", args)

    main(args)  # pragma: no cover
