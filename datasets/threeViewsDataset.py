import os
import torch
from PIL import Image

from torch.utils.data import Dataset

class ThreeViewsDataset(Dataset):
    def __init__(self, file_list, image_dir, transform=None, num_classes=5):
        self.file_list = file_list
        self.image_dir = image_dir
        self.transform = transform
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.file_list)

    def load_image_jpg_JPG(self, filename):
        img_path_jpg = os.path.join(self.image_dir, f"{filename}.jpg")
        img_path_JPG = os.path.join(self.image_dir, f"{filename}.JPG")

        # Check if either jpg or JPG file exists
        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        elif os.path.exists(img_path_JPG):
            img_path = img_path_JPG

        return img_path

    def width_center_crop(self, image, crop_ratio):
        width = image.width
        height = image.height
        crop_width = int(width * crop_ratio)
        left = (width - crop_width) // 2
        top = 0
        right = left + crop_width
        bottom = height
        
        return image.crop((left, top, right, bottom))
    
    def __getitem__(self, idx):
        base_name = self.file_list[idx][0]

        img_FA = self.load_image_jpg_JPG(f"{base_name}FA")
        img_SD = self.load_image_jpg_JPG(f"{base_name}SD")
        img_FP = self.load_image_jpg_JPG(f"{base_name}FP")
        
        image1 = Image.open(img_FA)
        image2 = Image.open(img_SD)
        image3 = Image.open(img_FP)

        image1 = self.width_center_crop(image1, 9/10)
        image2 = self.width_center_crop(image2, 2/3)
        image3 = self.width_center_crop(image3, 9/10)

        # Define the desired height
        desired_height = min(image1.height, image2.height, image3.height)
        
        # Calculate the new widths while maintaining the aspect ratio
        new_width1 = int((desired_height / image1.height) * image1.width)
        new_width2 = int((desired_height / image2.height) * image2.width)
        new_width3 = int((desired_height / image3.height) * image3.width)
        
        # Resize the images to have the same height
        image1 = image1.resize((new_width1, desired_height), Image.LANCZOS)
        image2 = image2.resize((new_width2, desired_height), Image.LANCZOS)
        image3 = image3.resize((new_width3, desired_height), Image.LANCZOS)

        # Create a new image with the calculated dimensions
        concatenated_image = Image.new("RGB", (new_width1 + new_width2 + new_width3, desired_height))

        # Paste the images side by side
        concatenated_image.paste(image1, (0, 0))
        concatenated_image.paste(image2, (new_width1, 0))
        concatenated_image.paste(image3, (new_width1 + new_width2, 0))
        
        if self.transform:
            image = self.transform(concatenated_image)

        label_to_index = {
            5:{
                'ecto': 0, 
                'ecto-meso': 1,
                'meso': 2,
                'meso-endo': 3,
                'endo': 4
            },
            3:{
                'ecto': 0, 
                'meso': 1,
                'endo': 2
            }
        }

        label = label_to_index[self.num_classes][self.file_list[idx][1]]
        
        return image, label