import logging
import numpy as np

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data.build import filter_images_with_only_crowd_annotations


def split_train_data(cfg):
    new_ds = []
    cfg.DATASETS.UNLABELED = list(cfg.DATASETS.UNLABELED)
    for name in cfg.DATASETS.TRAIN:
        if '_split_' in name:
            ds_name, num_split = name.split('_split_')
            labelled, unlabelled = split_dataset_labelled_unlabelled(
                ds_name,
                int(num_split),
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                seed=cfg.SEED)
            if labelled is not None:
                new_ds.append(labelled)
            if unlabelled is not None:
                cfg.DATASETS.UNLABELED.append(unlabelled)
        else:
            new_ds.append(name)
    cfg.DATASETS.TRAIN = new_ds
    return cfg


def split_dataset_labelled_unlabelled(dataset_name, num_labelled, filter_empty=True, seed=0):
    # get metadata
    metadata = MetadataCatalog.get(dataset_name)

    # get dataset and split indices
    dataset_dicts = DatasetCatalog.get(dataset_name)
    if filter_empty:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    num_instances = len(dataset_dicts)
    indices = np.arange(num_instances)
    np.random.seed(seed)
    np.random.shuffle(indices)
    labelled_indices = sorted(indices[:num_labelled])
    unlabelled_indices = sorted(indices[num_labelled:])

    #register datasets
    register_coco_instances_with_split(f"{dataset_name}_labelled", metadata.name, metadata.json_file, metadata.image_root, labelled_indices, filter_empty)
    if len(unlabelled_indices) > 0:
        register_coco_instances_with_split(f"{dataset_name}_unlabelled", metadata.name, metadata.json_file, metadata.image_root, unlabelled_indices, filter_empty)
        return f"{dataset_name}_labelled", f"{dataset_name}_unlabelled"
    else:
        return f"{dataset_name}_labelled", None


def register_coco_instances_with_split(name, parent, json_file, image_root, indices, filter_empty):
    if name in DatasetCatalog.keys():
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)
    DatasetCatalog.register(name, lambda: load_coco_json_with_split(json_file, image_root, parent, indices, filter_empty))
    metadata = MetadataCatalog.get(parent)
    MetadataCatalog.get(name).set(json_file=json_file, image_root=image_root, name=name, thing_classes=metadata.thing_classes, thing_dataset_id_to_contiguous_id=metadata.thing_dataset_id_to_contiguous_id)
    
    
def load_coco_json_with_split(json_file, image_root, parent_name, indices, filter_empty):
    dataset_dicts = load_coco_json(json_file, image_root, parent_name)
    if filter_empty:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    logger = logging.getLogger("detectron2")
    logger.info("aldi.helpers: Splitting off {} images of {} images in COCO format from {}: first 10 indices {}".format(len(indices), len(dataset_dicts), json_file, indices[:min(len(indices), 10)]))
    return [dataset_dicts[index] for index in indices]
