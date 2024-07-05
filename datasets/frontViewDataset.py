import os
import torch
from PIL import Image

from torch.utils.data import Dataset

class FrontViewDataset(Dataset):
    def __init__(self, file_list, image_dir, transform=None):
        self.file_list = file_list
        self.image_dir = image_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        base_name = self.file_list[idx][0]

        img_path_jpg = os.path.join(self.image_dir, f"{base_name}FA.jpg")
        img_path_JPG = os.path.join(self.image_dir, f"{base_name}FA.JPG")
        
        # Check if either jpg or JPG file exists
        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        elif os.path.exists(img_path_JPG):
            img_path = img_path_JPG
        
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        label_to_index = {
            'ecto': 0, 
            'ecto-meso': 1,
            'meso': 2,
            'meso-endo': 3,
            'endo': 4
            }

        label = label_to_index[self.file_list[idx][1]]
        
        return image, label