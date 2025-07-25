import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from detectron2.config import configurable
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.layers import cat
from detectron2.layers.wrappers import cross_entropy
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.box_regression import _dense_box_regression_loss
from detectron2.utils.registry import Registry
from fvcore.nn import smooth_l1_loss

from aldi.helpers import SaveIO, ManualSeed, ReplaceProposalsOnce, set_attributes
from aldi.pseudolabeler_saod import SparseStudentTeacherPseudoLabeler
from aldi.distill import DISTILLER_REGISTRY, Distiller

from torchviz import make_dot


    


@DISTILLER_REGISTRY.register()
class SparseStudentTeacherDistiller(Distiller):
    """Compute hard or soft distillation (based on config values) for Faster R-CNN based students/teachers.
    """

    def __init__(self, teacher, student, do_hard_cls=False, do_hard_obj=False, do_hard_rpn_reg=False, do_hard_roi_reg=False,
                 pseudo_label_threshold=0.8, pseudo_label_method="thresholding", saod_labeling_method="StudentTeacher"):
        set_attributes(self, locals())
        self.register_hooks()
        self.pseudo_labeler = SparseStudentTeacherPseudoLabeler(teacher, student, pseudo_label_threshold, threshold_method=pseudo_label_method, labeling_method=saod_labeling_method)

    @classmethod
    def from_config(cls, cfg, teacher, student):
        return SparseStudentTeacherDistiller(teacher, student,
                                             do_hard_cls=cfg.DOMAIN_ADAPT.DISTILL.HARD_ROIH_CLS_ENABLED,
                                             do_hard_obj=cfg.DOMAIN_ADAPT.DISTILL.HARD_OBJ_ENABLED,
                                             do_hard_rpn_reg=cfg.DOMAIN_ADAPT.DISTILL.HARD_RPN_REG_ENABLED,
                                             do_hard_roi_reg=cfg.DOMAIN_ADAPT.DISTILL.HARD_ROIH_REG_ENABLED,
                                             pseudo_label_threshold=cfg.DOMAIN_ADAPT.TEACHER.THRESHOLD,
                                             pseudo_label_method=cfg.DOMAIN_ADAPT.TEACHER.PSEUDO_LABEL_METHOD,
                                             saod_labeling_method=cfg.SAOD.LABELING_METHOD)

    def distill_enabled(self):
        return True

    def register_hooks(self):
        student_model = self.student.module if type(self.student) is DDP else self.student
        teacher_model = self.teacher.module if type(self.teacher) is DDP else self.teacher

        for param in teacher_model.parameters():
            param.requires_grad = False

        # Make sure seeds are the same for proposal sampling in teacher/student
        self.seeder = ManualSeed()
        teacher_model.roi_heads.register_forward_pre_hook(self.seeder)
        student_model.roi_heads.register_forward_pre_hook(self.seeder)
        
    def _distill_forward(self, weak_batched_inputs, strong_batched_inputs):
        # first, get hard pseudo labels -- this is done in place
        # even if not included in overall loss, we need them for RPN proposal sampling
        
        self.pseudo_labeler(weak_batched_inputs, strong_batched_inputs)
        
        self.seeder.reset_seed()

        # teacher might be in eval mode -- this is important for inputs/outputs aligning
        was_eval = not self.teacher.training
        if was_eval: 
            self.teacher.train()
            
        
        
        if False:
            dot = make_dot(standard_losses['loss_cls'], params=dict(self.student.named_parameters()))
            dot.render("computation_graph", format="png")

        if self.saod_labeling_method != "CoStudent":
            standard_losses = self.student(weak_batched_inputs)
        else:
            standard_losses = self.student(weak_batched_inputs)
            strong_losses = self.student(strong_batched_inputs) # weak pred denoised by teacher and merged with gt
            for k, v in standard_losses.items():
                standard_losses[k] = (v + strong_losses[k])/2

        # return to eval mode if necessary
        if was_eval: 
            self.teacher.eval()
            
        return standard_losses

    def __call__(self, weak_batched_inputs, strong_batched_inputs):
        losses = {}

        # Do a forward pass to get activations, and get hard pseudo-label losses if desired
        #for b in weak_batched_inputs:
        #    b['image'] = b['img_weak']
        hard_losses = self._distill_forward(weak_batched_inputs, strong_batched_inputs)
        loss_to_attr = {
            "loss_cls": self.do_hard_cls,
            "loss_rpn_cls": self.do_hard_obj,
            "loss_rpn_loc": self.do_hard_rpn_reg,
            "loss_box_reg": self.do_hard_roi_reg,
        }
        for k, v in hard_losses.items():
            if loss_to_attr.get(k, False):
                #v.requires_grad = True
                losses[k] = v
            else:
                # Need to add to standard losses so that the optimizer can see it
                losses[k] = v * 0.0

        return losses
    
