import glob
import json
import pprint

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.modeling import build_model
from detectron2.data.build import get_detection_dataset_dicts, build_detection_train_loader, build_detection_test_loader, DatasetMapper
from detectron2.data.samplers import InferenceSampler

#from adapteacher import add_ateacher_config

import aldi.datasets # register datasets with Detectron2
import aldi.model # register ALDI R-CNN model with Detectron2
import aldi.backbone # register ViT FPN backbone with Detectron2

# #from adapteacher.engine.trainer import ATeacherTrainer, BaselineTrainer

from model_selection.utils import load_model_weights

# hacky way to register
#from adapteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
#from adapteacher.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
#from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN
#from adapteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
#import adapteacher.data.datasets.builtin

#from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from PIL import Image

import sys, os
import time
from modelSeleTools_DAS.methodsDirectory2Fast import *
from modelSeleTools_DAS.utils import setup


def get_voc_2012_test_images(dirname):
    imageset_file = os.path.join(dirname, "ImageSets/Main/test.txt")
    image_dir = os.path.join(dirname, "JPEGImages")

    with open(imageset_file, "r") as file:
        image_names = file.read().strip().split()[:1000]
    
    dataset_dicts = []
    for idx, name in enumerate(image_names):
        image_path = os.path.join(image_dir, f"{name}.jpg")
        record = {}
        record["file_name"] = image_path
        record["image_id"] = idx
        record["width"], record["height"] = Image.open(image_path).size
        dataset_dicts.append(record)
    
    return dataset_dicts

DatasetCatalog.register("voc_2012_test_images", lambda: get_voc_2012_test_images("./datasets/VOC2012/"))
MetadataCatalog.get("voc_2012_test_images").set(thing_classes=[])



def calculateMultiFast(cfg, model_dirs: list, method_functions: list, repeat=1):
    #if cfg.SEMISUPNET.Trainer == 'ateacher':
    #    Trainer = ATeacherTrainer
    #elif cfg.SEMISUPNET.Trainer == 'baseline':
    #    Trainer = BaselineTrainer
    #else:
    #    raise ValueError("Trainer Name is not found.")
    
    #assert args.eval_only, "Only eval-only mode supported."
    #assert cfg.SEMISUPNET.Trainer == "ateacher"
    #assert len(method_functions) >= 1, "Score function needed."

    # dataloader
    # Build data_loader
    #dataset = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
    
    # Updated DAS - Replaced DAS dataloading with dataloading based on _test_loader_from_config
    debug_length = 500
    source_dataset = get_detection_dataset_dicts(cfg.DATASETS.TRAIN, filter_empty=False)#[:debug_length]
    dataloader_source = build_detection_test_loader(source_dataset, mapper=DatasetMapper(cfg, False), sampler=InferenceSampler(len(source_dataset)))
    test_dataset = get_detection_dataset_dicts(cfg.DATASETS.TEST[0], filter_empty=False)#[:debug_length]
    dataloader_target = build_detection_test_loader(test_dataset, mapper=DatasetMapper(cfg, False), sampler=InferenceSampler(len(test_dataset)))    
    
    """
    if cfg.DATASETS.TEST[0] == "Clipart1k_test":
        dataloader_names = {"source": "voc_2012_test_images",
                            "target": "Clipart1k_test"}
        dataloader_source = Trainer.build_test_loader(cfg, dataloader_names["source"])
        dataloader_target = Trainer.build_test_loader(cfg, dataloader_names["target"])
    elif cfg.DATASETS.TEST[0] == "cityscapes_foggy_val":
        dataloader_names = {"target": "cityscapes_foggy_val",
                            "source": "cityscapes_fine_instance_seg_val"}
        dataloader_source = Trainer.build_test_loader(cfg, dataloader_names["source"])
        dataloader_target = Trainer.build_test_loader(cfg, dataloader_names["target"])
        # (optional)
        print("using loaded dataloaders")
        
        import pickle
        # with open("./digits/dataloaders/cityscapes_val_source.pkl", "rb") as f:
        #     dataloader_source = pickle.load(f)
        # with open("./digits/dataloaders/cityscapes_val_target.pkl", "rb") as f:
        #     dataloader_target = pickle.load(f)
    elif cfg.DATASETS.TEST[0] == 'cityval':
        dataloader_names = {"target": "cityval",
                            "source": "sim10kval"}
        dataloader_source = Trainer.build_test_loader(cfg, dataloader_names["source"])
        dataloader_target = Trainer.build_test_loader(cfg, dataloader_names["target"])
    else:
        dataloader_source = Trainer.build_test_loader(cfg, cfg.DATASETS.TRAIN[0])
        dataloader_target = Trainer.build_test_loader(cfg, cfg.DATASETS.TEST[0])
"""

    METHODS_DICT = {"ground_truth": []}
    for method_function in method_functions:
        METHODS_DICT[method_function.__name__] = []
    
    all_results = {}
    for model_dir in model_dirs: 
        # [MAIN]
        all_results[os.path.basename(model_dir)] = {}
        for method_function in method_functions:
            # 1 model construction
            model = build_model(cfg)
            model.eval()
            model.training = False
            #model_teacher = Trainer.build_model(cfg)
            #ensem_ts_model = EnsembleTSModel(model_teacher, model)  # whole model with teacher and student.

            # loading parameters
            #DetectionCheckpointer(
            #    ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            #).resume_or_load(model_dir, resume=args.resume)
            load_model_weights(model_dir, model) # Loads Teacher (EMA) model if it's present.

            result_dict = method_function(cfg, 
                                          model, #ensem_ts_model.modelTeacher,
                                          [dataloader_source, dataloader_target],
                                          repeat)
            
            
            for key in list(result_dict.keys()):
                if key not in list(METHODS_DICT.keys()) and key != 'score':
                    METHODS_DICT[key] = [result_dict[key]]
                    all_results[os.path.basename(model_dir)][key] = result_dict[key]
                elif key != "score":
                    METHODS_DICT[key].append(result_dict[key])
                    all_results[os.path.basename(model_dir)][key] = result_dict[key]
                else:
                    METHODS_DICT[method_function.__name__].append(result_dict['score'])
                    all_results[os.path.basename(model_dir)][method_function.__name__] = result_dict['score']

            maxKeyLength = max(len(key) for key in METHODS_DICT.keys())
            print()
            for key in METHODS_DICT.keys():
                print(f"'{key:<{maxKeyLength}}'\t: {METHODS_DICT[key]},")
            time.sleep(5)

    return METHODS_DICT, all_results



def main(args):
    cfg = setup(args)
    #if cfg.SEMISUPNET.Trainer == "ateacher":
    #    Trainer = ATeacherTrainer
    #elif cfg.SEMISUPNET.Trainer == "baseline":
    #    Trainer = BaselineTrainer
    #else:
    #    raise ValueError("Trainer Name is not found.")

    #assert args.eval_only, "Only eval-only mode supported."
    #assert cfg.SEMISUPNET.Trainer == "ateacher"

    model_dirs = sorted(glob.glob(os.path.join(cfg.OUTPUT_DIR, 'model_0*99.pth')))    
    result, all_result_dict = calculateMultiFast(cfg, model_dirs, [FIS, PDR])
    
    # Save results dictionary
    all_result_dict['output_dir'] = cfg.OUTPUT_DIR
    all_result_dict['target_dataset'] = cfg.DATASETS.TEST[0]
    all_result_dict['source_dataset'] = cfg.DATASETS.TRAIN
    all_result_dict['config_file'] = args.config_file    
    with open(os.path.join(cfg.OUTPUT_DIR, 'DAS_outputs.json'), 'w') as file:
        json.dump(all_result_dict, file)
        
    # DAS - normalize measures and calculate DAS (sum of FIS and PDR)
    def normalize(array):
        normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
        return normalized_array.tolist()
    
    model_keys = [k for k in list(all_result_dict.keys()) if "model_" in k]
    for measure in ['PDR', 'FIS']:
        measures = [all_result_dict[mk][measure] for mk in model_keys]
        print(measure, measures)
        if measure == 'FIS':
            measures = [f[1][0]*-1 for f in measures]
        normalized_measures = normalize(np.array(measures))
        for i, mk in enumerate(model_keys):
            all_result_dict[mk][f"{measure}_normalized"] = normalized_measures[i]
    for mk in model_keys:
        all_result_dict[mk]["DAS"] = all_result_dict[mk]["FIS_normalized"] + all_result_dict[mk]["PDR_normalized"]
    
    # Save outputs to file
    pprint.pp(all_result_dict)
    with open(os.path.join(cfg.OUTPUT_DIR, 'DAS_outputs.json'), 'w') as file:
        json.dump(all_result_dict, file)
        
    #with open(os.path.join(cfg.OUTPUT_DIR, 'DAS_outputs.json'), 'r') as file:
    #    data = json.load(file)
    
    return all_result_dict


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )