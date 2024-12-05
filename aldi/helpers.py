import random
import itertools
import json
import numpy as np
import os


from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data.build import filter_images_with_only_crowd_annotations

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate


from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval

import logging


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
            self._logger.warn("aldi.helpers: No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "aldi.helpers: Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("aldi.helpers: Some metrics cannot be computed and is shown as NaN.")

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
        self._logger.info("aldi.helpers: Per-category {} AP: \n".format(iou_type) + table)

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
        self._logger.info("aldi.helpers: Per-category {} AP50: \n".format(iou_type) + table)

        results.update({"AP50-" + name: ap for name, ap in results_per_category})

        return results

class Detectron2COCOIOUEvaluatorAdapter(Detectron2COCOEvaluatorAdapter):
    """A COCOEvaluator that makes iscrowd & area optional."""

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("aldi.helpers: Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("aldi.helpers: Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("aldi.helpers: Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    cocoeval_fn=COCOeval_opt_IOU,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = {f'AP{int(coco_eval.params.iouThrs[0]*100)}': (coco_eval.stats[0])}

            self._results[task] = res

class COCOeval_opt_IOU(COCOeval_opt):
    """A COCOEvaluator that makes iscrowd & area optional."""
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        super().__init__(cocoGt=cocoGt, cocoDt=cocoDt, iouType=iouType)
        self.params.iouThrs = np.array([0.25])

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((1,))
            stats[0] = _summarize(1, iouThr=self.params.iouThrs[0], maxDets=self.params.maxDets[2])
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets

        self.stats = summarize()


def split_train_data(cfg):
    new_ds = []
    cfg.DATASETS.UNLABELED = list(cfg.DATASETS.UNLABELED)
    for name in cfg.DATASETS.TRAIN:
        if '_split_' in name:
            ds_name, num_split = name.split('_split_')
            labelled, unlabelled = split_dataset_labelled_unlabelled(ds_name, int(num_split), filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS)
            new_ds.append(labelled)
            cfg.DATASETS.UNLABELED.append(unlabelled)
        else:
            new_ds.append(name)
    cfg.DATASETS.TRAIN = new_ds
    return cfg

def split_dataset_labelled_unlabelled(dataset_name, num_labelled, filter_empty=True):
    # get metadata
    metadata = MetadataCatalog.get(dataset_name)

    # get dataset and split indices
    dataset_dicts = DatasetCatalog.get(dataset_name)
    if filter_empty:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    num_instances = len(dataset_dicts)
    indices = np.arange(num_instances)
    np.random.shuffle(indices)
    labelled_indices = indices[:num_labelled]
    unlabelled_indices = indices[num_labelled:]

    #register datasets
    register_coco_instances_with_split(f"{dataset_name}_labelled", metadata, metadata.json_file, metadata.image_root, labelled_indices, filter_empty)
    register_coco_instances_with_split(f"{dataset_name}_unlabelled", metadata, metadata.json_file, metadata.image_root, unlabelled_indices, filter_empty)
    return f"{dataset_name}_labelled", f"{dataset_name}_unlabelled"

def register_coco_instances_with_split(name, metadata, json_file, image_root, indices, filter_empty):
    DatasetCatalog.register(name, lambda: load_coco_json_with_split(json_file, image_root, metadata.name, indices, filter_empty))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", 
    )
def load_coco_json_with_split(json_file, image_root, parent_name, indices, filter_empty):
    dataset_dicts = load_coco_json(json_file, image_root, parent_name)
    if filter_empty:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    logger = logging.getLogger("detectron2")
    logger.info("aldi.helpers: Splitting off {} images of {} images in COCO format from {}".format(len(indices), len(dataset_dicts), json_file))
    return [dataset_dicts[index] for index in indices]