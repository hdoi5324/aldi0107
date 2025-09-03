import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from detectron2.structures import Boxes, pairwise_iou
from aldi.pseudolabeler import process_pseudo_label
from aldi.gaussian_rcnn.instances import FreeInstances as Instances 
from model_selection.utils import _bbox_overlaps as bbox_overlaps


class SparseStudentTeacherPseudoLabeler:
    def __init__(self, teacher_model, student_model, threshold, threshold_method, labeling_method="TeacherPred", 
                 alpha_1_threshold=0.5, alpha_2_threshold=0.9, denoise_priority="iou"):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.threshold = threshold
        self.threshold_method = threshold_method
        self.labeling_method = labeling_method
        self.alpha_1_threshold = alpha_1_threshold
        self.alpha_2_threshold = alpha_2_threshold
        self.denoise_priority = denoise_priority

    def __call__(self, sparse_labeled_weak, sparse_labeled_strong):
        if self.labeling_method == "TeacherPred":
            return sparse_student_teacher_pseudo_label_inplace(self.teacher_model, sparse_labeled_weak, sparse_labeled_strong, 
                                                               self.threshold, self.threshold_method)
        elif self.labeling_method == "StrongTeacherWWeakStudentW":
            return sparse_strong_teacherw_weak_studentw_pseudo_label_inplacev2(self.teacher_model, self.student_model, sparse_labeled_weak, sparse_labeled_strong, 
                                                         self.threshold, self.threshold_method, self.denoise_priority,
                                                         self.alpha_1_threshold, self.alpha_2_threshold)            
        elif self.labeling_method == "WeakTeacherWStrongStudentW":
            return sparse_weak_teacherw_strong_studentw_pseudo_label_inplacev3(self.teacher_model, self.student_model, sparse_labeled_weak, sparse_labeled_strong, 
                                                         self.threshold, self.threshold_method, self.denoise_priority,
                                                         self.alpha_1_threshold, self.alpha_2_threshold)  
        elif self.labeling_method == "StrongStudentWWeakStudentS":
            return sparse_strong_studentw_weak_students_pseudo_label_inplace(self.teacher_model, self.student_model, sparse_labeled_weak, sparse_labeled_strong, 
                                                         self.threshold, self.threshold_method, self.denoise_priority,
                                                         self.alpha_1_threshold, self.alpha_2_threshold)
        else:
            return "Not implemented"

def detach_predictions(list_of_instances):
    for item in list_of_instances:
        for k, v in item.get_fields().items():
            if k == "pred_boxes":
                v.tensor.detach()
            else:
                v.detach()
    return list_of_instances


def denoise_predictions_list(noisy_predictions, teacher_predictions, priority="iou", alpha_1_threshold=0.5):
    denoised_predictions = []
    for noisy_pred_per_image, teacher_pred_per_image in zip(noisy_predictions, teacher_predictions):
        denoised_instances = denoise_detections(teacher_pred_per_image, noisy_pred_per_image, 
                                                alpha_1_threshold=alpha_1_threshold, priority=priority)
        denoised_predictions.append(denoised_instances)
    return denoised_predictions


def denoise_detections(Instance1, Instance2, alpha_1_threshold=0.5, priority="iou"):
    '''
    Adapted from https://github.com/hustvl/CoStudent/blob/main/configs/CoStudent_fcos.res50.score.05.teahcer_score.06.cocomiss50/fcos.py
    Based on Instance1 to revise Instance2,
    we revise Instance2 by scores and ious when both of Instances have scores,
    Namely,(Instance1[pred_output1],Instance2[pred_output2]),depending on their scores and ious
    Selects instance1 with highest score if it's suitable to be used for refinement.
    '''
    assert Instance1.image_size == Instance2.image_size
    image_size = Instance1.image_size

    refine_gt_Instance = Instances(tuple(image_size))
    missing_Instance = Instances(tuple(image_size))

    bboxes1 = Instance1.gt_boxes.tensor
    scores1 = Instance1.scores
    classes1 = Instance1.gt_classes

    bboxes2 = Instance2.gt_boxes.tensor.clone()
    scores2 = Instance2.scores.clone()
    classes2 = Instance2.gt_classes.clone()

    ious = bbox_overlaps(bboxes1,bboxes2)

    if len(ious)==0 or len(ious[0])==0:
        return Instances.cat([Instance1,Instance2])

    iou_mask = ious > alpha_1_threshold
    scores_mask = scores1.unsqueeze(1) > scores2.unsqueeze(1).T
    refinement_candidates = scores_mask & iou_mask
    refine_gt_inds = refinement_candidates.any(dim=0)
    if priority == "iou":
        masked_ious = torch.where(refinement_candidates, ious, torch.tensor(0.0))
        refine_inds = masked_ious.argmax(dim=0)
    elif priority == "score":
        masked_scores = torch.where(refinement_candidates, scores1.unsqueeze(1).expand_as(ious), torch.tensor(0.0))
        refine_inds = masked_scores.argmax(dim=0)

    refine_inds = refine_inds[refine_gt_inds]
    refine_gt_inds = torch.where(refine_gt_inds)[0]
    refine_gt_inds_repeat = refine_gt_inds.reshape(-1,1).repeat(1,4)

    bboxes2.scatter_(dim=0,index=refine_gt_inds_repeat,src=bboxes1[refine_inds])
    classes2.scatter_(dim=0,index=refine_gt_inds,src=classes1[refine_inds])
    scores2.scatter_(dim=0,index=refine_gt_inds,src=scores1[refine_inds])

    refine_gt_Instance.gt_boxes = Boxes((bboxes2+bboxes2.abs())/2)
    refine_gt_Instance.gt_classes = classes2
    refine_gt_Instance.scores = scores2

    missing_inds = (ious<alpha_1_threshold).all(dim=1)
    missing_Instance.gt_boxes = Boxes(bboxes1[missing_inds])
    missing_Instance.gt_classes = classes1[missing_inds]
    missing_Instance.scores = scores1[missing_inds]

    return Instances.cat([missing_Instance,refine_gt_Instance])
@torch.no_grad()
def merge_ground_truth_costudent(targets, predictions, iou_threshold):
    """
    Adapted from https://github.com/hustvl/CoStudent/blob/main/configs/CoStudent_fcos.res50.score.05.teahcer_score.06.cocomiss50/fcos.py
    Take all targets (sparse) which are the ground truth and any predictions 
    that don't overlap well with target based on iou_threshold
    """
    for target_batch_per_image, predictions_per_image in zip(targets, predictions):
        targets_per_image = target_batch_per_image['instances']
        image_size = targets_per_image.image_size
    
        missing_Instance = Instances(tuple(image_size))
    
        bboxes1 = targets_per_image.gt_boxes.tensor
        classes1 = targets_per_image.gt_classes
    
        bboxes2 = predictions_per_image.gt_boxes.tensor.clone()
        classes2 = predictions_per_image.gt_classes.clone()

        ious = bbox_overlaps(bboxes1, bboxes2)
        iou_mask = ious > iou_threshold
        class_mask = classes1.unsqueeze(1) == classes2.unsqueeze(1).T
        combined_mask = iou_mask & class_mask
        
        missing_inds = torch.sum(combined_mask, 0) == 0
        missing_Instance.gt_boxes = Boxes(bboxes2[missing_inds])
        missing_Instance.gt_classes = classes2[missing_inds]
  
        target_batch_per_image['instances'] = Instances.cat([missing_Instance, targets_per_image])


def sparse_student_teacher_pseudo_label_inplace(model, labeled_weak, labeled_strong, 
                                                score_threshold, threshold_method, alpha_3_threshold=0.4):
    with torch.no_grad():
        # get predictions from teacher model on weakly-augmented data
        # do_postprocess=False to disable transforming outputs back into original image space
        was_training = model.training
        model.eval()
        teacher_preds = model.inference(labeled_weak, do_postprocess=False)
        if was_training: model.train()

        # postprocess pseudo labels (thresholding)
        teacher_preds, _ = process_pseudo_label(teacher_preds, score_threshold, threshold_method)
        merge_ground_truth_costudent(labeled_weak, teacher_preds, alpha_3_threshold)

        # add pseudo labels back as "ground truth" to labeled_strong
        for weak, strong in zip(labeled_weak, labeled_strong):
            strong["instances"] = weak['instances']


def sparse_strong_studentw_weak_students_pseudo_label_inplace(teacher_model, student_model, labeled_weak, labeled_strong, 
                                                              score_threshold, threshold_method, denoise_priority="iou", 
                                                              alpha_1_threshold=0.5, alpha_2_threshold=0.9, alpha_3_threshold=0.4, ):
    student_model = student_model.module if type(student_model) is DDP else student_model
    was_teacher_training = teacher_model.training
    teacher_model.eval()
    was_student_training = student_model.training
    student_model.eval()

    # Get predictions - lots of detaching from graph, probably too much.
    teacher_preds = detach_predictions(teacher_model.inference([{"image": item['image'].detach().clone()} for item in labeled_weak], do_postprocess=False))
    student_weak_preds = detach_predictions(student_model.inference([{"image": item['image'].detach().clone()} for item in labeled_weak], do_postprocess=False))
    student_strong_preds = detach_predictions(student_model.inference([{"image": item['image'].detach().clone()} for item in labeled_strong], do_postprocess=False))

    if was_teacher_training: teacher_model.train()    
    if was_student_training: student_model.train()
    
    # postprocess pseudo labels (thresholding)
    teacher_preds, _ = process_pseudo_label(teacher_preds, score_threshold, threshold_method)
    student_weak_preds, _ = process_pseudo_label(student_weak_preds, score_threshold, threshold_method)
    student_strong_preds, _ = process_pseudo_label(student_strong_preds, score_threshold, threshold_method)
    
    # Denoise student preds with teacher preds
    student_weak_preds = denoise_predictions_list(student_weak_preds, teacher_preds, priority=denoise_priority, alpha_1_threshold=alpha_1_threshold)
    student_strong_preds = denoise_predictions_list(student_strong_preds, teacher_preds, priority=denoise_priority, alpha_1_threshold=alpha_1_threshold)
    
    # Merge resulting preds with gt from opposite aug type eg weak preds with strong batch
    merge_ground_truth_costudent(labeled_weak, student_strong_preds, alpha_3_threshold)
    merge_ground_truth_costudent(labeled_strong, student_weak_preds, alpha_3_threshold)

def sparse_strong_teacherw_weak_studentw_pseudo_label_inplacev2(teacher_model, student_model, labeled_weak, labeled_strong, 
                                                                score_threshold, threshold_method, denoise_priority, 
                                                                alpha_1_threshold=0.5, alpha_2_threshold=0.9, alpha_3_threshold=0.4):
    student_model = student_model.module if type(student_model) is DDP else student_model
    was_teacher_training = teacher_model.training
    teacher_model.eval()
    was_student_training = student_model.training
    student_model.eval()

    teacher_preds = detach_predictions(teacher_model.inference([{"image": item['image'].detach().clone()} for item in labeled_weak], do_postprocess=False)) #todo: try copying labeled_weak
    student_weak_preds = detach_predictions(student_model.inference([{"image": item['image'].detach().clone()} for item in labeled_weak], do_postprocess=False))

    if was_teacher_training: teacher_model.train()    
    if was_student_training: student_model.train()
    
    # postprocess pseudo labels (thresholding)
    teacher_preds, _ = process_pseudo_label(teacher_preds, score_threshold, threshold_method)
    student_weak_preds, _ = process_pseudo_label(student_weak_preds, score_threshold, threshold_method)
    
    # Denoise student preds with teacher preds
    student_weak_preds = denoise_predictions_list(student_weak_preds, teacher_preds, priority=denoise_priority, alpha_1_threshold=alpha_1_threshold)
    
    # Merge resulting preds with gt from opposite aug type eg weak preds with strong batch
    merge_ground_truth_costudent(labeled_strong, teacher_preds, alpha_3_threshold)
    merge_ground_truth_costudent(labeled_weak, student_weak_preds, alpha_3_threshold)
    
def sparse_weak_teacherw_strong_studentw_pseudo_label_inplacev3(teacher_model, student_model, labeled_weak, labeled_strong, 
                                                                score_threshold, threshold_method, denoise_priority, 
                                                                alpha_1_threshold=0.5, alpha_2_threshold=0.9, alpha_3_threshold=0.4):
    student_model = student_model.module if type(student_model) is DDP else student_model
    was_teacher_training = teacher_model.training
    teacher_model.eval()
    was_student_training = student_model.training
    student_model.eval()

    teacher_preds = detach_predictions(teacher_model.inference([{"image": item['image'].detach().clone()} for item in labeled_weak], do_postprocess=False)) #todo: try copying labeled_weak
    student_weak_preds = detach_predictions(student_model.inference([{"image": item['image'].detach().clone()} for item in labeled_weak], do_postprocess=False))

    if was_teacher_training: teacher_model.train()    
    if was_student_training: student_model.train()
    
    # postprocess pseudo labels (thresholding)
    teacher_preds, _ = process_pseudo_label(teacher_preds, score_threshold, threshold_method)
    student_weak_preds, _ = process_pseudo_label(student_weak_preds, score_threshold, threshold_method)
    
    # Denoise student preds with teacher preds
    student_weak_preds = denoise_predictions_list(student_weak_preds, teacher_preds, priority=denoise_priority, alpha_1_threshold=alpha_1_threshold)
    
    # Merge resulting preds with gt from opposite aug type eg weak preds with strong batch
    merge_ground_truth_costudent(labeled_weak, teacher_preds, alpha_3_threshold)
    merge_ground_truth_costudent(labeled_strong, student_weak_preds, alpha_3_threshold)


