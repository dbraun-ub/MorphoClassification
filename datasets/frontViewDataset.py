import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset



class FrontViewDataset(Dataset):
    def __init__(self, file_list, image_dir, transform=None, num_classes=5):
        self.file_list = file_list
        self.image_dir = image_dir
        self.transform = transform
        self.num_classes = num_classes
    
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
    

class FrontViewDatasetV2(Dataset):
    def __init__(self, file_list, image_dir, size, transform=None, num_classes=5):
        self.file_list = file_list
        self.image_dir = image_dir
        self.transform = transform
        self.num_classes = num_classes
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
    
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
        else:
            print(f"Image not found: {filename}")

        return img_path
    
    @staticmethod
    def center_crop_resize(img, size, center):
        target_height, target_width = size
        aspect_ratio = target_width / target_height
        width, height = img.size
        
        target_width = width
        target_height = width // aspect_ratio
        
        if target_height > height:
            target_height = height
            target_width = int(height * aspect_ratio)

        c_x, c_y = center
        
        left = max(0, int(c_x - target_width / 2))
        top = max(0, int(c_y - target_height / 2))
        right = min(width, left + target_width)
        bottom = min(height, top + target_height)
        
        # Adjust left and top in case crop size is out of image bounds
        left = right - target_width
        top = bottom - target_height
        
        img_crop = img.crop((left, top, right, bottom))
        
        img_resize = img_crop.resize(size, Image.BILINEAR)

        return img_resize
    
    def __getitem__(self, idx):
        item = self.file_list.loc[idx]
        base_name = item['filename']

        img_path = self.load_image_jpg_JPG(f"{base_name}FA")
        
        image = Image.open(img_path)

        FA_markers = ["FA{:02d}".format(i) for i in [2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
        
        x = np.zeros(len(FA_markers))
        y = np.zeros(len(FA_markers))
        
        for idx, m in enumerate(FA_markers):
            x[idx] = item['x_pix_' + m]
            y[idx] = item['y_pix_' + m]

        image = self.center_crop_resize(image, self.size, (np.mean(x), np.mean(y)))

        if self.transform:
            image = self.transform(image)

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

        label = label_to_index[self.num_classes][item["morpho-1"]]
        
        return image, label