from detectron2.data.datasets import register_coco_instances

cyclegan_results_name_cityscapes2foggy = "cityscapes2foggy_002"
cyclegan_results_epoch_cityscapes2foggy = 50

# Cityscapes 
register_coco_instances("cityscapes_train", {"image_dir_prefix": "datasets/cityscapes/leftImg8bit/train", "translated_image_dir": f"datasets/daod-strong-baseline-cyclegan-results/{cyclegan_results_name_cityscapes2foggy}/test_{cyclegan_results_epoch_cityscapes2foggy}/images/fake_A"},         "datasets/cityscapes/annotations/cityscapes_train_instances.json",                  "datasets/cityscapes/leftImg8bit/train/")
register_coco_instances("cityscapes_val",   {},         "datasets/cityscapes/annotations/cityscapes_val_instances.json",                    "datasets/cityscapes/leftImg8bit/val/")

# Foggy Cityscapes
register_coco_instances("cityscapes_foggy_train", {"image_dir_prefix": "datasets/cityscapes/leftImg8bit_foggy/train", "translated_image_dir": f"datasets/daod-strong-baseline-cyclegan-results/{cyclegan_results_name_cityscapes2foggy}/test_{cyclegan_results_epoch_cityscapes2foggy}/images/fake_B"},   "datasets/cityscapes/annotations/cityscapes_train_instances_foggyALL.json",   "datasets/cityscapes/leftImg8bit_foggy/train/")
register_coco_instances("cityscapes_foggy_val", {},     "datasets/cityscapes/annotations/cityscapes_val_instances_foggyALL.json",     "datasets/cityscapes/leftImg8bit_foggy/val/")
# for evaluating COCO-pretrained models: category IDs are remapped to match
register_coco_instances("cityscapes_foggy_val_coco_ids", {},     "datasets/cityscapes/annotations/cityscapes_val_instances_foggyALL_coco.json",     "datasets/cityscapes/leftImg8bit_foggy/val/")

# Sim10k
register_coco_instances("sim10k_cars_train", {"image_dir_prefix": "datasets/sim10k/images", "translated_image_dir": f"datasets/daod-strong-baseline-cyclegan-results/sim10k2cityscapes/test_20/images/fake_A"},             "datasets/sim10k/coco_car_annotations.json",                  "datasets/sim10k/images/")
register_coco_instances("cityscapes_cars_train", {"image_dir_prefix": "datasets/cityscapes/leftImg8bit/train", "translated_image_dir": f"datasets/daod-strong-baseline-cyclegan-results/sim10k2cityscapes/test_20/images/fake_B"},         "datasets/cityscapes/annotations/cityscapes_train_instances_cars.json",                  "datasets/cityscapes/leftImg8bit/train/")
register_coco_instances("cityscapes_cars_val",   {},         "datasets/cityscapes/annotations/cityscapes_val_instances_cars.json",                    "datasets/cityscapes/leftImg8bit/val/")

# CFC
register_coco_instances("cfc_train", {"image_dir_prefix": "datasets/cfc/images/cfc_train", "translated_image_dir": f"datasets/daod-strong-baseline-cyclegan-results/cfc_002/test_1/images/fake_A"},         "datasets/cfc/coco_labels/cfc_train.json",                  "datasets/cfc/images/cfc_train/")
register_coco_instances("cfc_val",   {},         "datasets/cfc/coco_labels/cfc_val.json",                    "datasets/cfc/images/cfc_val/")
register_coco_instances("cfc_channel_train", {"image_dir_prefix": "datasets/cfc/images/cfc_channel_train", "translated_image_dir": f"datasets/daod-strong-baseline-cyclegan-results/cfc_002/test_1/images/fake_B"},         "datasets/cfc/coco_labels/cfc_channel_train.json",                  "datasets/cfc/images/cfc_channel_train/")
register_coco_instances("cfc_channel_test",   {},         "datasets/cfc/coco_labels/cfc_channel_test.json",                    "datasets/cfc/images/cfc_channel_test/")

# Urchin synthetic nudi_urchin3
register_coco_instances("nudi_urchin3_train", {}, "datasets/collated_outputs/nudi_urchin3/annotations/instances_train2023.json", "datasets/collated_outputs/nudi_urchin3/train2023")
register_coco_instances("nudi_urchin3_test", {}, "datasets/collated_outputs/nudi_urchin3/annotations/instances_test2023.json", "datasets/collated_outputs/nudi_urchin3/test2023")

# Urchin synthetic urchininf_v0
register_coco_instances("urchininf_v0_train", {}, "datasets/collated_outputs/urchininf_v0/annotations/instances_train2023_urchin.json", "datasets/collated_outputs/urchininf_v0/train2023")
register_coco_instances("urchininf_v0_train_all", {}, "datasets/collated_outputs/urchininf_v0/annotations/instances_train2023_all.json", "datasets/collated_outputs/urchininf_v0/train2023")
register_coco_instances("urchininf_v0_test", {}, "datasets/collated_outputs/urchininf_v0/annotations/instances_test2023_urchin.json", "datasets/collated_outputs/urchininf_v0/test2023")
register_coco_instances("urchininf_v0_test_all", {}, "datasets/collated_outputs/urchininf_v0/annotations/instances_test2023_all.json", "datasets/collated_outputs/urchininf_v0/test2023")

# Urchin synthetic urchininf_rov_v1
register_coco_instances("urchininf_rov_v1_train", {}, "datasets/collated_outputs/urchininf_rov_v1/annotations/instances_train2023_urchin.json", "datasets/collated_outputs/urchininf_rov_v1/train2023")
register_coco_instances("urchininf_rov_v1_train_all", {}, "datasets/collated_outputs/urchininf_rov_v1/annotations/instances_train2023_all.json", "datasets/collated_outputs/urchininf_rov_v1/train2023")
register_coco_instances("urchininf_rov_v1_test", {}, "datasets/collated_outputs/urchininf_rov_v1/annotations/instances_test2023_urchin.json", "datasets/collated_outputs/urchininf_rov_v1/test2023")
register_coco_instances("urchininf_rov_v1_test_all", {}, "datasets/collated_outputs/urchininf_rov_v1/annotations/instances_test2023_all.json", "datasets/collated_outputs/urchininf_rov_v1/test2023")

# Urchin synthetic urchininf_auv_v2
register_coco_instances("urchininf_auv_v2_train", {}, "datasets/collated_outputs/nudi_urchin_auv_v2/annotations/instances_train2023_urchin.json", "datasets/collated_outputs/nudi_urchin_auv_v2/train2023")
register_coco_instances("urchininf_auv_v2_train_all", {}, "datasets/collated_outputs/nudi_urchin_auv_v2/annotations/instances_train2023_all.json", "datasets/collated_outputs/nudi_urchin_auv_v2/train2023")
register_coco_instances("urchininf_auv_v2_test", {}, "datasets/collated_outputs/nudi_urchin_auv_v2/annotations/instances_test2023_urchin.json", "datasets/collated_outputs/nudi_urchin_auv_v2/test2023")
register_coco_instances("urchininf_auv_v2_test_all", {}, "datasets/collated_outputs/nudi_urchin_auv_v2/annotations/instances_test2023_all.json", "datasets/collated_outputs/nudi_urchin_auv_v2/test2023")
register_coco_instances("nudi_handfish_auv_v1_train", {}, "datasets/collated_outputs/nudi_handfish_auv_v1/annotations/instances_train2023.json", "datasets/collated_outputs/nudi_handfish_auv_v1/train2023")
register_coco_instances("nudi_handfish_auv_v1_test", {}, "datasets/collated_outputs/nudi_handfish_auv_v1/annotations/instances_test2023.json", "datasets/collated_outputs/nudi_handfish_auv_v1/test2023")
register_coco_instances("nudi_handfish_auv_v1_train_nowater", {}, "datasets/collated_outputs/nudi_handfish_auv_v1/annotations/instances_train2023.json", "datasets/collated_outputs/nudi_handfish_auv_v1/train2023_nowater")
register_coco_instances("nudi_handfish_auv_v1_test_nowater", {}, "datasets/collated_outputs/nudi_handfish_auv_v1/annotations/instances_test2023.json", "datasets/collated_outputs/nudi_handfish_auv_v1/test2023_nowater")

register_coco_instances("nudi_handfish_auv_v2_train", {}, "datasets/collated_outputs/nudi_handfish_auv_v2/annotations/instances_train2023.json", "datasets/collated_outputs/nudi_handfish_auv_v2/train2023")
register_coco_instances("nudi_handfish_auv_v2_test", {}, "datasets/collated_outputs/nudi_handfish_auv_v2/annotations/instances_test2023.json", "datasets/collated_outputs/nudi_handfish_auv_v2/test2023")
register_coco_instances("nudi_handfish_auv_v2_train_nowater", {}, "datasets/collated_outputs/nudi_handfish_auv_v2/annotations/instances_train2023.json", "datasets/collated_outputs/nudi_handfish_auv_v2/train2023_nowater")
register_coco_instances("nudi_handfish_auv_v2_test_nowater", {}, "datasets/collated_outputs/nudi_handfish_auv_v2/annotations/instances_test2023.json", "datasets/collated_outputs/nudi_handfish_auv_v2/test2023_nowater")

register_coco_instances("nudi_handfish_rov_v3_train", {}, "datasets/collated_outputs/nudi_handfish_rov_v3/annotations/instances_train2023.json", "datasets/collated_outputs/nudi_handfish_rov_v3/train2023")
register_coco_instances("nudi_handfish_rov_v3_test", {}, "datasets/collated_outputs/nudi_handfish_rov_v3/annotations/instances_test2023.json", "datasets/collated_outputs/nudi_handfish_rov_v3/test2023")
register_coco_instances("nudi_handfish_rov_v3_train_nowater", {}, "datasets/collated_outputs/nudi_handfish_rov_v3/annotations/instances_train2023.json", "datasets/collated_outputs/nudi_handfish_rov_v3/train2023_nowater")
register_coco_instances("nudi_handfish_rov_v3_test_nowater", {}, "datasets/collated_outputs/nudi_handfish_rov_v3/annotations/instances_test2023.json", "datasets/collated_outputs/nudi_handfish_rov_v3/test2023_nowater")

register_coco_instances("trench_handfish_auv_v1_train", {}, "datasets/collated_outputs/trench_handfish_auv_v1/annotations/instances_train2023.json", "datasets/collated_outputs/trench_handfish_auv_v1/train2023")
register_coco_instances("trench_handfish_auv_v1_test", {}, "datasets/collated_outputs/trench_handfish_auv_v1/annotations/instances_test2023.json", "datasets/collated_outputs/trench_handfish_auv_v1/test2023")
register_coco_instances("trench_handfish_auv_v1_train_nowater", {}, "datasets/collated_outputs/trench_handfish_auv_v1/annotations/instances_train2023.json", "datasets/collated_outputs/trench_handfish_auv_v1/train2023_nowater")
register_coco_instances("trench_handfish_auv_v1_test_nowater", {}, "datasets/collated_outputs/trench_handfish_auv_v1/annotations/instances_test2023.json", "datasets/collated_outputs/trench_handfish_auv_v1/test2023_nowater")


# UDD
register_coco_instances("UDD_test", {}, "datasets/UDD/annotations/instances_test2023_remap.json", "datasets/UDD/test2023")
register_coco_instances("UDD_train", {}, "datasets/UDD/annotations/instances_train2023_remap.json", "datasets/UDD/train2023")
register_coco_instances("UDD_train_without_target", {}, "datasets/UDD/annotations/instances_train2023_remap_without_target.json", "datasets/UDD/train2023")
register_coco_instances("UDD_train_with_target", {}, "datasets/UDD/annotations/instances_train2023_remap_with_target.json", "datasets/UDD/train2023")

# Squidle
register_coco_instances("squidle_urchin_2011_test", {}, "datasets/squidle_coco/squidle_urchin_2011/annotations/instances_test2023.json", "datasets/squidle_coco/squidle_urchin_2011/test2023")
register_coco_instances("squidle_urchin_2009_train", {}, "datasets/squidle_coco/squidle_urchin_2009/annotations/instances_train2023.json", "datasets/squidle_coco/squidle_urchin_2009/train2023")
register_coco_instances("squidle_urchin_2009_train_split100", {}, "datasets/squidle_coco/squidle_urchin_2009/annotations/instances_train2023_split100.json", "datasets/squidle_coco/squidle_urchin_2009/train2023")
register_coco_instances("squidle_urchin_2009_train_without_target", {}, "datasets/squidle_coco/squidle_urchin_2009/annotations/instances_train2023_without_target.json", "datasets/squidle_coco/squidle_urchin_2009/train2023")
register_coco_instances("squidle_urchin_2009_train_with_target", {}, "datasets/squidle_coco/squidle_urchin_2009/annotations/instances_train2023_with_target.json", "datasets/squidle_coco/squidle_urchin_2009/train2023")
register_coco_instances("squidle_east_tas_urchins_train", {}, "datasets/squidle_coco/squidle_east_tas_urchins/annotations/instances_train2023.json", "datasets/squidle_coco/squidle_east_tas_urchins/train2023")
register_coco_instances("squidle_east_tas_urchins_train_without_target", {}, "datasets/squidle_coco/squidle_east_tas_urchins/annotations/instances_train2023_without_target.json", "datasets/squidle_coco/squidle_east_tas_urchins/train2023")
register_coco_instances("squidle_east_tas_urchins_train_with_target", {}, "datasets/squidle_coco/squidle_east_tas_urchins/annotations/instances_train2023_with_target.json", "datasets/squidle_coco/squidle_east_tas_urchins/train2023")

register_coco_instances("squidle_pretrain_train", {}, "datasets/squidle_coco/handfish_iros/squidle_handfish_pretrain/annotations/instances_train2023.json", "datasets/squidle_coco/handfish_iros/squidle_handfish_pretrain/train2023")
register_coco_instances("squidle_pretrain_test", {}, "datasets/squidle_coco/handfish_iros/squidle_handfish_pretrain/annotations/instances_test2023.json", "datasets/squidle_coco/handfish_iros/squidle_handfish_pretrain/test2023")

#Squidle handfish
register_coco_instances("sq_hand_test15v2", {}, "datasets/squidle_coco/sq_hand_test15v2/annotations/instances_test2023.json", "datasets/squidle_coco/sq_hand_test15v2/test2023")
register_coco_instances("sq_hand_train85_n200v2", {}, "datasets/squidle_coco/sq_hand_train85_n200v2/annotations/instances_train2023.json", "datasets/squidle_coco/sq_hand_train85_n200v2/train2023")
register_coco_instances("squidle_handfish_15800_train", {}, "datasets/squidle_coco/squidle_handfish_15800/annotations/instances_train2023.json", "datasets/squidle_coco/squidle_handfish_15800/train2023")
register_coco_instances("squidle_handfish_15800_test", {}, "datasets/squidle_coco/squidle_handfish_15800/annotations/instances_test2023.json", "datasets/squidle_coco/squidle_handfish_15800/test2023")
register_coco_instances("squidle_handfish_16511_train", {}, "datasets/squidle_coco/squidle_handfish_16511/annotations/instances_train2023.json", "datasets/squidle_coco/squidle_handfish_16511/train2023")


# S-UODAC2020
register_coco_instances("SUODAC2020_test", {}, "datasets/S-UODAC2020/COCO_Annotations/instances_target.json", "datasets/S-UODAC2020/test")
register_coco_instances("SUODAC2020_train", {}, "datasets/S-UODAC2020/COCO_Annotations/instances_source.json", "datasets/S-UODAC2020/train")


