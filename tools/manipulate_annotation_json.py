import os
import json

from pycocotools.coco import COCO


def main():
    #ann_files = ["datasets/squidle_coco/squidle_urchin_2009/annotations/instances_train2023.json", "datasets/squidle_coco/squidle_east_tas_urchins/annotations/instances_train2023.json"]
    ann_files = ["datasets/collated_outputs/urchininf_v0/annotations/instances_train2023_urchin.json", "datasets/collated_outputs/urchininf_rov_v1/annotations/instances_train2023_urchin.json", "datasets/collated_outputs/nudi_urchin_auv_v2/annotations/instances_train2023_urchin.json"]
    ann_files = ["datasets/collated_outputs/urchininf_v0/annotations/instances_test2023_urchin.json", "datasets/collated_outputs/urchininf_rov_v1/annotations/instances_test2023_urchin.json", "datasets/collated_outputs/nudi_urchin_auv_v2/annotations/instances_test2023_urchin.json"]

    #ann_files = ["datasets/UDD/annotations/instances_train2023_remap.json"]

    coco_files = [COCO(f) for f in ann_files]

    #for f in ann_files:
    #    split_file(f, COCO(f))

    combined_dataset = combine_coco(coco_files)
    new_combined_file = "/home/heather/GitHub/aldi0107/datasets/collated_outputs/synthetic_urchin/annotations/instances_test2017.json"
    with open(new_combined_file, "w") as fp:
        json.dump(combined_dataset, fp)
    print(f"Saved {new_combined_file}")
    
    ## Separate with and without target
    #description = "Based on squidle_urchin_2009. Split by images with and without annotations"
    #coco = coco_files[0]
    #separate_images_with_target(ann_files[0], coco, description)


def combine_coco(coco_files):
    new_id, new_ann_id = 0, 0
    new_images, new_annotations = [], []
    for coco_file in coco_files:
        for img_id, img in coco_file.imgs.items():
            anns = coco_file.imgToAnns[img_id]
            if len(anns) > 0:
                for a in anns:
                    a['image_id'] = new_id
                    a['id'] = new_ann_id
                    new_annotations.append(a)
                    new_ann_id += 1
                img['id'] = new_id
                new_images.append(img)
                new_id += 1
    
    return {'info': coco_files[0].dataset.get('info', {}),
            'categories': coco_files[0].dataset['categories'],
            'images': new_images,
            'annotations': new_annotations}



def split_file(ann_file, coco, split_no=100, description=""):
    no_new_datasets = len(coco.imgs) // split_no + 1
    new_datasets = []
    json_dir, json_file = os.path.split(ann_file)
    for n in range(no_new_datasets):
        new_imgs = coco.dataset["images"][n * split_no:(n + 1) * split_no]
        new_anns = [coco.imgToAnns[img['id']] for img in new_imgs]
        new_anns = sum(new_anns, [])
        categories = [{'id': 1, 'name': 'seaurchin', 'supercategory': 'benthic'}]
        new_dataset = {'info': coco.dataset.get('info', {}),
                'categories': categories, #coco.dataset['categories'],
                'images': new_imgs,
                'annotations': new_anns}
        new_dataset['info']['categories'] = categories 
        new_datasets.append(new_dataset)


        new_file = os.path.join(json_dir,
                                f"{json_file[:-5]}_split{n}.json")
        with open(new_file, "w") as fp:
            json.dump(new_dataset, fp)
        print(f"Saved {new_file}")

    return new_datasets





if __name__ == "__main__":
    main()
