import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from detectron2.structures.boxes import Boxes
# from detectron2.structures.instances import Instances
from aldi.gaussian_rcnn.instances import FreeInstances as Instances # TODO: only when necessary
from model_selection.utils import _bbox_overlaps

from detectron2.structures import Boxes, pairwise_iou
from aldi.pseudolabeler import process_pseudo_label
from model_selection.utils import _bbox_overlaps as bbox_overlaps


class SparseStudentTeacherPseudoLabeler:
    def __init__(self, teacher_model, student_model, threshold, threshold_method, labeling_method="StudentTeacher", alpha_1_threshold=0.5, alpha_2_threshold=0.9):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.threshold = threshold
        self.threshold_method = threshold_method
        self.labeling_method = labeling_method
        self.alpha_1_threshold = alpha_1_threshold
        self.alpha_2_threshold = alpha_2_threshold

    def __call__(self, sparse_labeled_weak, sparse_labeled_strong):
        if self.labeling_method == "StudentTeacher":
            return sparse_student_teacher_pseudo_label_inplace(self.teacher_model, sparse_labeled_weak, sparse_labeled_strong, self.threshold, self.threshold_method)
        elif self.labeling_method == "CoStudent":
            return sparse_costudent_pseudo_label_inplace(self.teacher_model, self.student_model, sparse_labeled_weak, sparse_labeled_strong, 
                                                         self.threshold, self.threshold_method,
                                                         self.alpha_1_threshold, self.alpha_2_threshold)
        else:
            return "Not implemented"

def sparse_student_teacher_pseudo_label_inplace(model, labeled_weak, labeled_strong, score_threshold, threshold_method, alpha_3_threshold=0.4):
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

def detach_predictions(list_of_instances):
    for item in list_of_instances:
        for k, v in item.get_fields().items():
            if k == "pred_boxes":
                v.tensor.detach()
            else:
                v.detach()
    return list_of_instances


@torch.no_grad()
def merge_ground_truth_costudent(targets, predictions, iou_thresold):
    
    for target_batch_per_image, predictions_per_image in zip(targets, predictions):
        targets_per_image = target_batch_per_image['instances']
        image_size = targets_per_image.image_size

        predictions_per_image_cvt = predictions_per_image

        iou_matrix = pairwise_iou(targets_per_image.gt_boxes,
                                  predictions_per_image_cvt.gt_boxes)
        iou_filter = iou_matrix > iou_thresold

        target_class_list = (targets_per_image.gt_classes).reshape(-1, 1)
        pred_class_list = (predictions_per_image_cvt.gt_classes).reshape(1, -1)
        class_filter = target_class_list == pred_class_list

        final_filter = iou_filter & class_filter
        unlabel_idxs = torch.sum(final_filter, 0) == 0

        new_target = Instances(image_size)
        new_target.gt_boxes = Boxes.cat([targets_per_image.gt_boxes,
                                         predictions_per_image_cvt.gt_boxes[unlabel_idxs]])
        new_target.gt_classes = torch.cat([targets_per_image.gt_classes,
                                           predictions_per_image_cvt.gt_classes[unlabel_idxs]])
        
        target_batch_per_image['instances'] = new_target


def sparse_costudent_pseudo_label_inplace(teacher_model, student_model, labeled_weak, labeled_strong, score_threshold, threshold_method, 
                                          alpha_1_threshold=0.5, alpha_2_threshold=0.9, alpha_3_threshold=0.4):
    student_model = student_model.module if type(student_model) is DDP else student_model
    was_teacher_training = teacher_model.training
    teacher_model.eval()
    was_student_training = student_model.training
    student_model.eval()

    teacher_preds = detach_predictions(teacher_model.inference([{"image": item['image'].detach().clone()} for item in labeled_weak], do_postprocess=False)) #todo: try copying labeled_weak
    student_weak_preds = detach_predictions(student_model.inference([{"image": item['image'].detach().clone()} for item in labeled_weak], do_postprocess=False))
    #student_strong_preds = detach_predictions(student_model.inference([{"image": item['image'].detach().clone()} for item in labeled_strong], do_postprocess=False))

    #teacher_preds = detach_predictions(teacher_model.inference([{"image": item['image']} for item in labeled_weak], do_postprocess=False)) #todo: try copying labeled_weak
    #student_weak_preds = detach_predictions(student_model.inference([{"image": item['image']} for item in labeled_weak], do_postprocess=False))
    #student_strong_preds = detach_predictions(student_model.inference([{"image": item['image']} for item in labeled_strong], do_postprocess=False))

    if was_teacher_training: teacher_model.train()    
    if was_student_training: student_model.train()
    
    # postprocess pseudo labels (thresholding)
    teacher_preds, _ = process_pseudo_label(teacher_preds, score_threshold, threshold_method)
    student_weak_preds, _ = process_pseudo_label(student_weak_preds, score_threshold, threshold_method)
    #student_strong_preds, _ = process_pseudo_label(student_strong_preds, score_threshold, threshold_method)
    
    # Denoise student preds with teacher preds
    student_weak_preds = denoise_predictions_costudent(student_weak_preds, teacher_preds, alpha_1_threshold=alpha_1_threshold)
    #student_strong_preds = denoise_predictions_costudent(student_strong_preds, teacher_preds, alpha_1_threshold=alpha_1_threshold)
    
    # Merge resulting preds with gt from opposite aug type eg weak preds with strong batch
    #merge_ground_truth_costudent(labeled_weak, student_strong_preds, alpha_3_threshold)
    merge_ground_truth_costudent(labeled_weak, teacher_preds, alpha_3_threshold)
    merge_ground_truth_costudent(labeled_strong, student_weak_preds, alpha_3_threshold)


def denoise_predictions_costudent(noisy_predictions, teacher_predictions, alpha_1_threshold=0.5):
    denoised_predictions = []
    for noisy_pred_per_image, teacher_pred_per_image in zip(noisy_predictions, teacher_predictions):
        denoised_instances = Revision_PRED(noisy_pred_per_image, teacher_pred_per_image)
        denoised_predictions.append(denoised_instances)
    return denoised_predictions


def denoise_preds(noisy_predictions, teacher_predictions, alpha_1_threshold=0.5, alpha_2_threshold=0.9):
    denoised_predictions = []
    for noisy_pred, teacher_pred in zip(noisy_predictions, teacher_predictions):
        denoised_pred = []
        # Select the best of the noisy_predictions.  Replace with GT prediction if needed.
        for j in range(len(noisy_pred)):
            noisy_instance = noisy_pred[j]
            noisy_class = noisy_instance.gt_classes[0].item()
            noisy_score = noisy_instance.scores[noisy_class].item()
            candidate_instances = []
            for i in range(len(teacher_pred)):
                teacher_instance = teacher_pred[i]
                overlap = pairwise_iou(teacher_instance.gt_boxes[0], noisy_instance.gt_boxes[0]).item()
                teacher_class = teacher_instance.gt_classes[0].item()
                teacher_score = teacher_instance.scores[teacher_class].item()
                if overlap >= alpha_1_threshold and noisy_class == teacher_class and teacher_score > noisy_score:
                    candidate_instances.append(teacher_instance)
                elif overlap >= alpha_1_threshold and noisy_class == teacher_class and teacher_score <= noisy_score:
                    candidate_instances.append(noisy_instance)
                elif overlap > alpha_2_threshold and noisy_class != teacher_class and teacher_score > noisy_score:
                    candidate_instances.append(teacher_instance)
                else:
                    candidate_instances.append(noisy_instance)
            if len(candidate_instances) > 0:
                max_instance = max(candidate_instances, key=lambda instance: instance.scores[0].item())    
                denoised_pred.append(max_instance)
            
        # Add teacher instances that don't overlap with any chosen instances.
        for i in range(len(teacher_pred)):
            overlap_with_teacher = False
            for chosen_instance in denoised_pred:
                if pairwise_iou(chosen_instance.gt_boxes[0], teacher_pred[i].gt_boxes[0]) > alpha_1_threshold:
                    overlap_with_teacher = True
                    break
            if not overlap_with_teacher:
                denoised_pred.append(teacher_pred[i])
        if len(denoised_pred) == 0:
            denoised_predictions.append(noisy_pred)
        else:
            denoised_predictions.append(denoised_pred[0].cat(denoised_pred))
    return denoised_predictions


def Revision_PRED(Instance1, Instance2, alpha_1_threshold=0.5, alpha_2_threshold=0.9):
    '''
    Based on Instance1 to revise Instance2,
    we revise Instance2 by scores and ious when both of Instances have scores,
    Namely,(Instance1[pred_output1],Instance2[pred_output2]),depending on their scores and ious

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

    # Iterate through each candidate Instance1 for each Instance2 to see if it is the best replacement (refinement)
    while(True):

        refine_gt_inds = (ious > alpha_1_threshold).any(dim=0)
        #refine_inds = ious.max(dim=0)[1]
        
        mask = ious > alpha_1_threshold
        masked_scores = torch.where(mask, scores1.unsqueeze(1).expand_as(ious), torch.tensor(-1.0))
        refine_inds = masked_scores.argmax(dim=0)
        
        refine_gt_scores = scores1[refine_inds]
        need_refine = refine_gt_scores >= scores2

        lower_scores_inds = ~need_refine & refine_gt_inds

        lower_scores_inds0 = torch.where(lower_scores_inds)[0]

        if lower_scores_inds0.numel()>0:
            index = [refine_inds[lower_scores_inds],lower_scores_inds0]

            input_zeros = torch.zeros((lower_scores_inds0.numel())).to(ious.device) + alpha_1_threshold

            ious.index_put_(index,input_zeros)

        else:
            break

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


def merge_sparse_gt_with_preds_in_place(sparse_labeled_batch, predictions, threshold):
    """sparse_labeled_batch has the sparse gt predictions for the batch.  
    Based on CoStudent merge logic.
    Merge GT with the predictions made using these rules:
    Use all the GT boxes
    If no GT overlaps with a prediciton, keep it.
    If a prediction overlaps with GT and it's a different class, keep it."""
    
    for labeled_item, predictions in zip(sparse_labeled_batch, predictions):
        # copy og fields in sparse just in case.
        sparse_instance = labeled_item['instances']
        idx_to_keep = []
        for pred_idx in range(len(predictions)):
            pred_box = predictions[pred_idx].gt_boxes[0]
            overlap_with_gt = False
            for gt_idx in range(len(sparse_instance)):
                gt_box = sparse_instance[gt_idx].gt_boxes[0]
                overlap = pairwise_iou(pred_box, gt_box).item()
                if overlap > threshold:
                    gt_class = sparse_instance[gt_idx].gt_classes[0].item()
                    pred_class = predictions[pred_idx].gt_classes[0].item()
                    if pred_class != gt_class:
                        idx_to_keep.append(pred_idx)
                    else:
                        overlap_with_gt = True

            if not overlap_with_gt:
                idx_to_keep.append(pred_idx)

        merged_instance = sparse_instance.cat([sparse_instance] + [predictions[i] for i in idx_to_keep])
        labeled_item['instances'] = merged_instance

