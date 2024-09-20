import copy
import random
import torchvision
from detectron2.data import detection_utils as utils
import torch
import detectron2.data.transforms as T
import numpy as np
from detectron2.data import DatasetCatalog
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes

"""Adapted from https://github.com/PacktPublishing/Hands-On-Computer-Vision-with-Detectron2/blob/main/Chapter09/Detectron2_Chapter09_CustomAugmentations.ipynb"""

def generate_mosaics(labelled_multiimg_aug_data):
    # get three more random images
    mo_items = labelled_multiimg_aug_data
    # images
    mosaic_data = []
    no_mosaics = len(mo_items)//4
    for i in range(no_mosaics):
        # Get four images and corresponding boxes
        dataset_dicts = mo_items[i*4:i*4+4]
        boxes = []
        imgs, weak_imgs = [], []
        classes = []
        for ds in dataset_dicts:
            imgs.append(ds["image"])
            weak_imgs.append(ds["img_weak"])
            boxes.append(ds["instances"].get("gt_boxes"))
            classes.append(ds["instances"].get("gt_classes"))
        mt = MosaicTransform(imgs, boxes)
        mt_weak = MosaicTransform(weak_imgs, None)
        image = mt.apply_image()
        weak_img = mt_weak.apply_image()
        orig_h, orig_w = mt.loc_info[0], mt.loc_info[1]
        new_h, new_w = orig_h//2, orig_w//2

        boxes = mt.apply_box()
        crop_y, crop_x = random.randint(0, orig_h-new_h), random.randint(0, orig_w-new_w)
        boxes.tensor -= torch.tensor([new_h, new_w, new_h, new_w])
        boxes.clip((new_h, new_w))
        mask = torch.nonzero(boxes.area() > 0)
        image = image[:,crop_y:crop_y+new_h,crop_x:crop_x+new_w]
        weak_img = weak_img[:,crop_y:crop_y+new_h,crop_x:crop_x+new_w]
        key_ds = dataset_dicts[0]
        key_ds["image"] = image
        key_ds["img_weak"] = weak_img
        key_ds["height"] = image.shape[1]
        key_ds["width"] = image.shape[2]
        if len(mask) == 0:
            key_ds["instances"] = Instances(tuple(image.shape[1:]),
                                            gt_boxes=Boxes(np.zeros((0, 4))),
                                            gt_classes=torch.tensor([], dtype=torch.int64))
            #print("No ground truth in mosaic")
        else:
            mask = mask.reshape((len(mask),))
            boxes = boxes[mask]
            classes = torch.cat(classes)[mask]
            key_ds["instances"] = Instances(tuple(image.shape[1:]), gt_boxes=boxes, gt_classes=classes)
        mosaic_data.append(key_ds)
    return mosaic_data



class MosaicTransform(T.Transform):
    def __init__(self, mo_images, mo_boxes):
        self.mo_images = mo_images
        self.mo_boxes = mo_boxes

    def get_loc_info(self):
        images = self.mo_images
        heights = [i.shape[1] for i in images]
        widths = [i.shape[2] for i in images]
        ch = max(heights[0], heights[1])
        cw = max(widths[0], widths[2])
        h = (max(heights[0], heights[1]) +
             max(heights[2], heights[3]))
        w = (max(widths[0], widths[2]) +
             max(widths[1], widths[3]))
        # pad or start coordinates
        y0, x0 = ch - heights[0], cw - widths[0]
        y1, x1 = ch - heights[1], cw
        y2, x2 = ch, cw - widths[2]
        y3, x3 = ch, cw
        x_pads = [x0, x1, x2, x3]
        y_pads = [y0, y1, y2, y3]
        return (h, w, ch, cw, widths, heights, x_pads, y_pads)

    def apply_image(self):
        # get the loc info
        self.loc_info = self.get_loc_info()
        h, w, ch, cw, widths, heights, x_pads, y_pads = self.loc_info
        if len(x_pads) < 4:
            print("not enough padding")
        # output
        images = self.mo_images
        output = torch.zeros((3, h, w), dtype=images[0].dtype, device=images[0].device)
        for i, img in enumerate(images):
            output[:, y_pads[i]: y_pads[i] + heights[i],
            x_pads[i]: x_pads[i] + widths[i]] = img

        return output

    def apply_coords(self, coords):
        return coords

    def apply_box(self):
        # combine boxes
        boxes = copy.deepcopy(self.mo_boxes)
        new_boxes = []
        # now update location values
        _, _, _, _, _, _, x_pads, y_pads = self.loc_info
        for i, bbox in enumerate(boxes):
            new_bbox = torch.add(bbox.tensor, torch.tensor([x_pads[i], y_pads[i], x_pads[i], y_pads[i]]))
            new_boxes.append(new_bbox)
        # flatten it
        key_box = boxes[0]
        key_box.tensor = torch.vstack(new_boxes)
        return key_box
