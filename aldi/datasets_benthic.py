from detectron2.data.datasets import register_coco_instances

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
register_coco_instances("squidle_urchin_train", {}, "datasets/squidle_coco/squidle_urchin_full_train/annotations/instances_train.json", "datasets/squidle_coco/squidle_urchin_full_train/images")
register_coco_instances("squidle_urchin_train_sparse", {}, "datasets/squidle_coco/squidle_urchin_full_train_sparse/annotations/instances_train.json", "datasets/squidle_coco/squidle_urchin_full_train_sparse/images")

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

# Transformed
register_coco_instances("urchininf_v0_train_transformed0", {}, "datasets/collated_outputs/urchininf_v0/annotations/instances_train2023_urchin.json", "datasets/collated_outputs/urchininf_v0/train2023_transformed_0")
register_coco_instances("urchininf_rov_v1_train_transformed0", {}, "datasets/collated_outputs/urchininf_rov_v1/annotations/instances_train2023_urchin.json", "datasets/collated_outputs/urchininf_rov_v1/train2023_transformed_0")
register_coco_instances("urchininf_auv_v2_train_transformed0", {}, "datasets/collated_outputs/nudi_urchin_auv_v2/annotations/instances_train2023_urchin.json", "datasets/collated_outputs/nudi_urchin_auv_v2/train2023_transformed_0")
register_coco_instances("urchininf_v0_train_transformed1", {}, "datasets/collated_outputs/urchininf_v0/annotations/instances_train2023_urchin.json", "datasets/collated_outputs/urchininf_v0/train2023_transformed_1")
register_coco_instances("urchininf_rov_v1_train_transformed1", {}, "datasets/collated_outputs/urchininf_rov_v1/annotations/instances_train2023_urchin.json", "datasets/collated_outputs/urchininf_rov_v1/train2023_transformed_1")
register_coco_instances("urchininf_auv_v2_train_transformed1", {}, "datasets/collated_outputs/nudi_urchin_auv_v2/annotations/instances_train2023_urchin.json", "datasets/collated_outputs/nudi_urchin_auv_v2/train2023_transformed_1")
register_coco_instances("urchininf_v0_train_transformed2", {}, "datasets/collated_outputs/urchininf_v0/annotations/instances_train2023_urchin.json", "datasets/collated_outputs/urchininf_v0/train2023_transformed_2")
register_coco_instances("urchininf_rov_v1_train_transformed2", {}, "datasets/collated_outputs/urchininf_rov_v1/annotations/instances_train2023_urchin.json", "datasets/collated_outputs/urchininf_rov_v1/train2023_transformed_2")
register_coco_instances("urchininf_auv_v2_train_transformed2", {}, "datasets/collated_outputs/nudi_urchin_auv_v2/annotations/instances_train2023_urchin.json", "datasets/collated_outputs/nudi_urchin_auv_v2/train2023_transformed_2")

register_coco_instances("sim10k_cars_train_transformed0", {"image_dir_prefix": "datasets/sim10k/images", "translated_image_dir": f"datasets/daod-strong-baseline-cyclegan-results/sim10k2cityscapes/test_20/images/fake_A"},             "datasets/sim10k/coco_car_annotations.json",                  "datasets/sim10k/images_transformed_0/")
register_coco_instances("sim10k_cars_train_transformed1", {"image_dir_prefix": "datasets/sim10k/images", "translated_image_dir": f"datasets/daod-strong-baseline-cyclegan-results/sim10k2cityscapes/test_20/images/fake_A"},             "datasets/sim10k/coco_car_annotations.json",                  "datasets/sim10k/images_transformed_1/")
register_coco_instances("sim10k_cars_train_transformed2", {"image_dir_prefix": "datasets/sim10k/images", "translated_image_dir": f"datasets/daod-strong-baseline-cyclegan-results/sim10k2cityscapes/test_20/images/fake_A"},             "datasets/sim10k/coco_car_annotations.json",                  "datasets/sim10k/images_transformed_2/")

# Redcup clip bbox from mass points  all then 1-2
register_coco_instances("loose_redcup17647_train", {}, "datasets/loose/loose_17647/annotations/instances_train.json", "datasets/loose/loose_17647/images")
register_coco_instances("loose_redcup17647_train_erased_only", {}, "datasets/loose/loose_17647/annotations/instances_train_erased_only.json", "datasets/loose/loose_17647/images")
register_coco_instances("loose_redcup17647_train_cropped_only", {}, "datasets/loose/loose_17647/annotations/instances_train_cropped_only.json", "datasets/loose/loose_17647/images")
register_coco_instances("loose_redcup17647_train_og_only", {}, "datasets/loose/loose_17647/annotations/instances_train_og_only.json", "datasets/loose/loose_17647/images")
register_coco_instances("loose_redcup17647_train_noclip", {}, "datasets/loose/loose_17647/annotations/instances_train_noclip.json", "datasets/loose/loose_17647/images")
register_coco_instances("loose_redcup17648_train", {}, "datasets/loose/loose_17648/annotations/instances_train.json", "datasets/loose/loose_17648/images")
register_coco_instances("loose_redcup17648_train_erased_only", {}, "datasets/loose/loose_17648/annotations/instances_train_erased_only.json", "datasets/loose/loose_17648/images")
register_coco_instances("loose_redcup17648_train_cropped_only", {}, "datasets/loose/loose_17648/annotations/instances_train_cropped_only.json", "datasets/loose/loose_17648/images")
register_coco_instances("loose_redcup17648_train_og_only", {}, "datasets/loose/loose_17648/annotations/instances_train_og_only.json", "datasets/loose/loose_17648/images")
register_coco_instances("loose_redcup17648_train_noclip", {}, "datasets/loose/loose_17648/annotations/instances_train_noclip.json", "datasets/loose/loose_17648/images")

# Redcup 1-2 points from bbox training set
register_coco_instances("loose_redcup17711_train", {}, "datasets/loose/loose_17711/annotations/instances_train.json", "datasets/loose/loose_17711/images")
register_coco_instances("loose_redcup17711_train_erased_only", {}, "datasets/loose/loose_17711/annotations/instances_train_erased_only.json", "datasets/loose/loose_17711/images")
register_coco_instances("loose_redcup17711_train_cropped_only", {}, "datasets/loose/loose_17711/annotations/instances_train_cropped_only.json", "datasets/loose/loose_17711/images")
register_coco_instances("loose_redcup17711_train_og_only", {}, "datasets/loose/loose_17711/annotations/instances_train_og_only.json", "datasets/loose/loose_17711/images")
register_coco_instances("loose_redcup17711_train_noclip", {}, "datasets/loose/loose_17711/annotations/instances_train_noclip.json", "datasets/loose/loose_17711/images")

register_coco_instances("squidle_redcup_test", {}, "datasets/squidle_coco/squidle_redcup_test/annotations/instances_test.json", "datasets/squidle_coco/squidle_redcup_test/test2023")
register_coco_instances("squidle_redcup_train", {}, "datasets/squidle_coco/squidle_redcup_full_train/annotations/instances_train.json", "datasets/squidle_coco/squidle_redcup_full_train/images")
register_coco_instances("squidle_redcup_train_sparse", {}, "datasets/squidle_coco/squidle_redcup_full_train_sparse/annotations/instances_train.json", "datasets/squidle_coco/squidle_redcup_full_train_sparse/images")

# Mass point annotation datasets
register_coco_instances("loose_urchin17630_train", {}, "datasets/loose/loose_17630/annotations/instances_train.json", "datasets/loose/loose_17630/images")
register_coco_instances("loose_urchin17630_train_erased_only", {}, "datasets/loose/loose_17630/annotations/instances_train_erased_only.json", "datasets/loose/loose_17630/images")
register_coco_instances("loose_urchin17630_train_cropped_only", {}, "datasets/loose/loose_17630/annotations/instances_train_cropped_only.json", "datasets/loose/loose_17630/images")
register_coco_instances("loose_urchin17630_train_og_only", {}, "datasets/loose/loose_17630/annotations/instances_train_og_only.json", "datasets/loose/loose_17630/images")
register_coco_instances("loose_urchin17630_train_noclip", {}, "datasets/loose/loose_17630/annotations/instances_train_noclip.json", "datasets/loose/loose_17630/images")
register_coco_instances("loose_urchin17631_train", {}, "datasets/loose/loose_17631/annotations/instances_train.json", "datasets/loose/loose_17631/images")
register_coco_instances("loose_urchin17631_train_erased_only", {}, "datasets/loose/loose_17631/annotations/instances_train_erased_only.json", "datasets/loose/loose_17631/images")
register_coco_instances("loose_urchin17631_train_cropped_only", {}, "datasets/loose/loose_17631/annotations/instances_train_cropped_only.json", "datasets/loose/loose_17631/images")
register_coco_instances("loose_urchin17631_train_og_only", {}, "datasets/loose/loose_17631/annotations/instances_train_og_only.json", "datasets/loose/loose_17631/images")
register_coco_instances("loose_urchin17631_train_noclip", {}, "datasets/loose/loose_17631/annotations/instances_train_noclip.json", "datasets/loose/loose_17631/images")

# Urchin 1-2 points with corresponding bbox annotated dataset
register_coco_instances("loose_urchin17714_train", {}, "datasets/loose/loose_17714/annotations/instances_train.json", "datasets/loose/loose_17714/images")
register_coco_instances("loose_urchin17714_train_erased_only", {}, "datasets/loose/loose_17714/annotations/instances_train_erased_only.json", "datasets/loose/loose_17714/images")
register_coco_instances("loose_urchin17714_train_cropped_only", {}, "datasets/loose/loose_17714/annotations/instances_train_cropped_only.json", "datasets/loose/loose_17714/images")
register_coco_instances("loose_urchin17714_train_og_only", {}, "datasets/loose/loose_17714/annotations/instances_train_og_only.json", "datasets/loose/loose_17714/images")
register_coco_instances("loose_urchin17714_train_noclip", {}, "datasets/loose/loose_17714/annotations/instances_train_noclip.json", "datasets/loose/loose_17714/images")

# Sparse coco 
register_coco_instances("coco_val", {}, "datasets/coco/annotations/instances_val2017.json", "datasets/coco/val2017")
register_coco_instances("coco_sparse_easy", {}, "datasets/coco/annotations/instances_train2017_easy.json", "datasets/coco/train2017")
register_coco_instances("coco_sparse_hard", {}, "datasets/coco/annotations/instances_train2017_hard.json", "datasets/coco/train2017")
register_coco_instances("coco_sparse_extreme", {}, "datasets/coco/annotations/instances_train2017_extreme.json", "datasets/coco/train2017")
register_coco_instances("coco_sparse_keep1", {}, "datasets/coco/annotations/keep1_instances_train2017.json", "datasets/coco/train2017")

register_coco_instances("squidle_yellowball_test", {}, "datasets/squidle_coco/squidle_yellowball_test/annotations/instances_test.json", "datasets/squidle_coco/squidle_yellowball_test/test2023")
