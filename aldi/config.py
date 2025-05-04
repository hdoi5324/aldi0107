# Everything in this file makes domain adaptation disabled by default.
# Everything must be explicitly enabled in the config files.

from detectron2.config import CfgNode as CN


def add_aldi_config(cfg):
    _C = cfg

    # Datasets and sampling
    _C.DATASETS.UNLABELED = tuple()
    _C.DATASETS.BATCH_CONTENTS = ("labeled_weak", ) # one or more of: { "labeled_weak", "labeled_strong", "unlabeled_weak", "unlabeled_strong" }
    _C.DATASETS.BATCH_RATIOS = (1,) # must match length of BATCH_CONTENTS
    _C.DATASETS.TRAIN_SIZE = 10e6 # Use to limit number of training images

    # Strong augmentations
    _C.AUG = CN()
    _C.AUG.WEAK_INCLUDES_MULTISCALE = True
    _C.AUG.LABELED_INCLUDE_RANDOM_ERASING = True
    _C.AUG.UNLABELED_INCLUDE_RANDOM_ERASING = True
    _C.AUG.LABELED_MIC_AUG = False
    _C.AUG.UNLABELED_MIC_AUG = False
    _C.AUG.MIC_RATIO = 0.5
    _C.AUG.MIC_BLOCK_SIZE = 32
    _C.AUG.MOSAIC = False
    _C.AUG.MOSAIC_P = 0.25

    # EMA of student weights
    _C.EMA = CN()
    _C.EMA.ENABLED = False
    _C.EMA.ALPHA = 0.9996
     # when loading a model at the start of training (i.e. not resuming mid-training run),
     # if MODEL.WEIGHTS contains both ["model", "ema"], initialize with the EMA weights.
     # also determines if EMA is used for eval when running tools/train_net.py --eval-only.
    _C.EMA.LOAD_FROM_EMA_ON_START = True

    # Begin domain adaptation settings
    _C.DOMAIN_ADAPT = CN()

    # Source-target alignment
    _C.DOMAIN_ADAPT.ALIGN = CN()
    _C.DOMAIN_ADAPT.ALIGN.IMG_DA_ENABLED = False
    _C.DOMAIN_ADAPT.ALIGN.IMG_DA_LAYER = "p2"
    _C.DOMAIN_ADAPT.ALIGN.IMG_DA_WEIGHT = 0.01
    _C.DOMAIN_ADAPT.ALIGN.IMG_DA_INPUT_DIM = 256 # = output channels of backbone
    _C.DOMAIN_ADAPT.ALIGN.IMG_DA_HIDDEN_DIMS = [256,]
    _C.DOMAIN_ADAPT.ALIGN.IMG_DA_IMPL = "ours" # {ours, at}
    _C.DOMAIN_ADAPT.ALIGN.INS_DA_ENABLED = False
    _C.DOMAIN_ADAPT.ALIGN.INS_DA_WEIGHT = 0.01
    _C.DOMAIN_ADAPT.ALIGN.INS_DA_INPUT_DIM = 1024 # = output channels of box head
    _C.DOMAIN_ADAPT.ALIGN.INS_DA_HIDDEN_DIMS = [1024,]

    # Self-distillation
    _C.DOMAIN_ADAPT.DISTILL = CN()
    _C.DOMAIN_ADAPT.DISTILL.DISTILLER_NAME = "ALDIDistiller"
    # 'Pseudo label' approaches
    _C.DOMAIN_ADAPT.DISTILL.HARD_ROIH_CLS_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.HARD_ROIH_REG_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.HARD_OBJ_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.HARD_RPN_REG_ENABLED = False
    # 'Distillation' approaches
    _C.DOMAIN_ADAPT.DISTILL.ROIH_CLS_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.ROIH_REG_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.OBJ_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.RPN_REG_ENABLED = False
    _C.DOMAIN_ADAPT.DISTILL.CLS_TMP = 1.0
    _C.DOMAIN_ADAPT.DISTILL.OBJ_TMP = 1.0
    _C.DOMAIN_ADAPT.CLS_LOSS_TYPE = "CE" # one of: { "CE", "KL" }

    # Teacher model provides pseudo labels
    # TODO: Could be merged into DISTILL settings somehow
    _C.DOMAIN_ADAPT.TEACHER = CN()
    _C.DOMAIN_ADAPT.TEACHER.ENABLED = False
    _C.DOMAIN_ADAPT.TEACHER.THRESHOLD = 0.8
    _C.DOMAIN_ADAPT.TEACHER.PSEUDO_LABEL_METHOD = "thresholding" # one of: { "thresholding", "probabilistic" }

    # Vision Transformer settings
    _C.VIT = CN()
    _C.VIT.USE_ACT_CHECKPOINT = True

    # We interpret SOLVER.IMS_PER_BATCH as the total batch size on all GPUs, for 
    # experimental consistency. Gradient accumulation is used according to 
    # num_gradient_accum_steps = IMS_PER_BATCH / (NUM_GPUS * IMS_PER_GPU)
    _C.SOLVER.IMS_PER_GPU = 2

    # Unbiased Mean Teacher style feature alignment
    _C.MODEL.UMT = CN()
    _C.MODEL.UMT.ENABLED = False
    
    # Fix issue with unused paratemetsr
    _C.MODEL.FIND_UNUSED_PARAMETERS = False

    # We use gradient accumulation to run the weak/strong/unlabeled data separately
    # Should we call backward intermittently during accumulation or at the end?
    # The former is slower but less memory usage
    _C.SOLVER.BACKWARD_AT_END = True

    # Enable use of different optimizers (necessary to match VitDet settings)
    _C.SOLVER.OPTIMIZER = "SGD"

    # to reproduce MIC
    _C.DOMAIN_ADAPT.ALIGN.SADA_ENABLED = False
    _C.DOMAIN_ADAPT.ALIGN.SADA_IMG_GRL_WEIGHT = 0.01
    _C.DOMAIN_ADAPT.ALIGN.SADA_INS_GRL_WEIGHT = 0.1
    _C.DOMAIN_ADAPT.ALIGN.SADA_COS_WEIGHT = 0.1
    _C.DOMAIN_ADAPT.LOSSES = CN()
    _C.DOMAIN_ADAPT.LOSSES.QUALITY_LOSS_WEIGHT_ENABLED = False

    # to reproduce PT (incomplete)
    _C.GRCNN = CN()
    _C.GRCNN.LEARN_ANCHORS_LABELED = False
    _C.GRCNN.LEARN_ANCHORS_UNLABELED = False
    _C.GRCNN.TAU = [0.5, 0.5]
    _C.GRCNN.EFL = False
    _C.GRCNN.EFL_LAMBDA = [0.5, 0.5]
    _C.GRCNN.MODEL_TYPE = "GAUSSIAN"

    # Neptune logging
    _C.LOGGING = CN()
    _C.LOGGING.PROJECT = "login/project_code"
    _C.LOGGING.API_TOKEN = "xxxxx"
    _C.LOGGING.ITERS = 100
    _C.LOGGING.TAGS = ""
    _C.LOGGING.GROUP_TAGS = ""
    _C.LOGGING.PROXY = ""

    # For unlabeled data do not remove any images as we don't know whether they have annotations.
    _C.DATALOADER.FILTER_UNLABELED_EMPTY_ANNOTATIONS = False

    _C.MODEL_SELECTION = CN()
    _C.MODEL_SELECTION.N_PERTURBATIONS = 3
    _C.MODEL_SELECTION.SCORE_THRESHOLD = 0.05
    _C.MODEL_SELECTION.PERTURB_TYPE = "dropout" # "dropout" OR "DAS"
    _C.MODEL_SELECTION.DROPOUT = 0.1
    _C.MODEL_SELECTION.SAMPLE = 250
    
    # In later Detectron2
    _C.DATALOADER.REPEAT_SQRT = True 
    _C.FLOAT32_PRECISION = ''
