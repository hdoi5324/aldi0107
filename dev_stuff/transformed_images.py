import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os

import torch
from torchvision.transforms import v2

# Define the transformations
transform_list = [
    v2.RandomApply(transforms=[v2.Grayscale()], p=0.8), 
    v2.ColorJitter(brightness=.3, contrast=0.2, saturation=0.1, hue=.3), 
    v2.RandomEqualize()
]


# Directories
input_dirs = ['/home/heather/GitHub/aldi0107/datasets/collated_outputs/nudi_urchin_auv_v2/train2023',
              '/home/heather/GitHub/aldi0107/datasets/collated_outputs/urchininf_rov_v1/train2023',
              '/home/heather/GitHub/aldi0107/datasets/collated_outputs/urchininf_v0/train2023',
              '/home/heather/GitHub/aldi0107/datasets/sim10k/images'
              ]

for input_dir in input_dirs:
    for i in range(3):
        transform = transforms.Compose([
            transform_list[i],
            transforms.ToTensor()
        ])
        output_dir, image_dir = os.path.split(input_dir)
        output_dir = os.path.join(output_dir, f"{image_dir}_transformed_{i}")
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Writing {image_dir} to {output_dir} with transforms {transform_list[i]}")
        
        # Process each image in the input directory
        for filename in tqdm(os.listdir(input_dir)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                # Open the image
                img_path = os.path.join(input_dir, filename)
                image = Image.open(img_path)
        
                # Apply the transformations
                transformed_image = transform(image)
        
                # Convert the tensor back to a PIL image
                transformed_image = transforms.ToPILImage()(transformed_image)
        
                # Save the transformed image to the output directory
                output_path = os.path.join(output_dir, filename)
                transformed_image.save(output_path)

print("Image transformation complete!")