import os
import torch
from PIL import Image

from torch.utils.data import Dataset

class ThreeViewsDataset(Dataset):
    def __init__(self, file_list, image_dir, transform=None):
        self.file_list = file_list
        self.image_dir = image_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        base_name = self.file_list[idx][0]
        img_paths = [os.path.join(self.image_dir, f"{base_name}{i}") for i in ['FA', 'SD', 'FP']]
        
        images = [Image.open(img_path) for img_path in img_paths]
        if self.transform:
            images = [self.transform(image) for image in images]
        
        # Concatener les images
        images = torch.cat(images, dim=0)

        label_to_index = {
            'ecto': 0, 
            'ecto-meso': 1,
            'meso': 2,
            'meso-endo': 3,
            'endo': 4
            }

        label = label_to_index[self.file_list[idx][1]]
        
        return images, label