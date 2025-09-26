import torch
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from torchvision.models.detection.fcos import FCOS as _FCOS
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.ops import boxes as box_ops, generalized_box_iou_loss, misc as misc_nn_ops, sigmoid_focal_loss
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.fcos import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights

__all__ = ["FCOSTorchvision"]


@META_ARCH_REGISTRY.register()
class FCOSTorchvision(_FCOS):
    """
    torchvision FCOS
    """

    def __init__(self, cfg):
        backbone = resnet50(weights=None, progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        backbone = _resnet_fpn_extractor(
            backbone, 3, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
        )
        super().__init__(
            backbone=backbone,
            num_classes=cfg.MODEL.FCOS.NUM_CLASSES,
            # transform parameters
            min_size=min(cfg.INPUT.MIN_SIZE_TRAIN),
            max_size=cfg.INPUT.MAX_SIZE_TRAIN,
            image_mean=cfg.MODEL.PIXEL_MEAN,
            image_std=cfg.MODEL.PIXEL_STD,
            # Anchor parameters
            anchor_generator=None,
            head=None,
            center_sampling_radius=cfg.MODEL.FCOS.POS_RADIUS,
            score_thresh=cfg.MODEL.FCOS.SCORE_THRESH_TEST,
            nms_thresh=cfg.MODEL.FCOS.NMS_THRESH_TEST,
            detections_per_img=cfg.TEST.DETECTIONS_PER_IMAGE,
            topk_candidates=cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST)
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))
        
        weights_dict = FCOS_ResNet50_FPN_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
        del weights_dict['head.classification_head.cls_logits.weight']
        del weights_dict['head.classification_head.cls_logits.bias']
        self.load_state_dict(weights_dict, strict=False)

    @classmethod
    def from_config(cls, cfg):
        return {"cfg": cfg}

    @property
    def device(self):
        return self.pixel_mean.device

    def inference(self, batched_inputs, do_postprocess=False):
        return self.forward(batched_inputs)

    def forward(self, batched_inputs, do_align=False, labeled=None):
        images = [item['image'] for item in batched_inputs]

        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            elif "targets" in batched_inputs[0]:
                gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            results, losses = self._forward(images, gt_instances)
            return losses
        else:
            results, losses = self._forward(images, None)
            processed_results = self._postprocess(results, batched_inputs, images.image_sizes)
            return processed_results 
        
    def _forward(self, images, targets):
        super().forward(images, targets)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images