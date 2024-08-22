import random
import itertools
import numpy as np

import torch
from tabulate import tabulate

from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import create_small_table


class SaveIO:
    """Simple PyTorch hook to save the output of a nn.module."""
    def __init__(self):
        self.input = None
        self.output = None
        
    def __call__(self, module, module_in, module_out):
        self.input = module_in
        self.output = module_out

class ManualSeed:
    """PyTorch hook to manually set the random seed."""
    def __init__(self):
        self.reset_seed()

    def reset_seed(self):
        self.seed = random.randint(0, 2**32 - 1)

    def __call__(self, module, args):
        torch.manual_seed(self.seed)

class ReplaceProposalsOnce:
    """PyTorch hook to replace the proposals with the student's proposals, but only once."""
    def __init__(self):
        self.proposals = None

    def set_proposals(self, proposals):
        self.proposals = proposals

    def __call__(self, module, args):
        ret = None
        if self.proposals is not None and module.training:
            images, features, proposals, gt_instances = args
            ret = (images, features, self.proposals, gt_instances)
            self.proposals = None
        return ret

def set_attributes(obj, params):
    """Set attributes of an object from a dictionary."""
    if params:
        for k, v in params.items():
            if k != "self" and not k.startswith("_"):
                setattr(obj, k, v)

class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight*grad_input, None

def grad_reverse(x):
    return _GradientScalarLayer.apply(x, -1.0)

def _maybe_add_optional_annotations(cocoapi) -> None:
    for ann in cocoapi.dataset["annotations"]:
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0
        if "area" not in ann:
            ann["area"] = ann["bbox"][1]*ann["bbox"][2]

class Detectron2COCOEvaluatorAdapter(COCOEvaluator):
    """A COCOEvaluator that makes iscrowd & area optional."""
    def __init__(
        self,
        dataset_name,
        output_dir=None,
        distributed=True,
    ):
        super().__init__(dataset_name, output_dir=output_dir, distributed=distributed)
        _maybe_add_optional_annotations(self._coco_api)


    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})

        # Now results per category for AP50
        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[0, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP50: \n".format(iou_type) + table)

        results.update({"AP50-" + name: ap for name, ap in results_per_category})

        return results

