import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
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
        else:
            print(f"Image not found: {filename}")

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
    


class ThreeViewsDatasetV2(Dataset):
    def __init__(self, file_list, image_dir, size, transform=None, num_classes=5):
        self.file_list = file_list
        self.image_dir = image_dir
        self.transform = transform
        self.num_classes = num_classes
        if isinstance(size, tuple):
            self.size = size
            if size[0] != size[1]:
                Warning('Works better with square input image.')
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
    def center_crop_resize(img, size, markers, item):
        x = np.zeros(len(markers))
        # y = np.zeros(len(markers))
        
        for idx, m in enumerate(markers):
            x[idx] = item['x_pix_' + m]
            # y[idx] = item['y_pix_' + m]

        c_x = (np.max(x) + np.min(x)) / 2
        
        target_height, target_width = size
        aspect_ratio = target_width / target_height
        width, height = img.size
        c_y = height // 2
        
        target_width = width
        target_height = width // aspect_ratio
        
        if target_height > height:
            target_height = height
            target_width = int(height * aspect_ratio)
        
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

    def concatenate_three_views(self, image1, image2, image3):
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

        # if self.transform:
        #     image1 = self.transform(image1)
        #     image2 = self.transform(image2)
        #     image3 = self.transform(image3)
        #     topil = transforms.ToPILImage()
        #     image1 = topil(image1)
        #     image2 = topil(image2)
        #     image3 = topil(image3)

        # Create a new image with the calculated dimensions
        concatenated_image = Image.new("RGB", (new_width1 + new_width2 + new_width3, desired_height))

        # Paste the images side by side
        concatenated_image.paste(image1, (0, 0))
        concatenated_image.paste(image2, (new_width1, 0))
        concatenated_image.paste(image3, (new_width1 + new_width2, 0))
        
        if self.transform:
            image = self.transform(concatenated_image)
        else:
            image = concatenated_image

        return image
        

    
    def __getitem__(self, idx):
        item = self.file_list.loc[idx]
        base_name = item['filename']

        img_path_FA = self.load_image_jpg_JPG(f"{base_name}FA")
        img_path_SD = self.load_image_jpg_JPG(f"{base_name}SD")
        img_path_FP = self.load_image_jpg_JPG(f"{base_name}FP")

        
        img_FA = Image.open(img_path_FA)
        img_SD = Image.open(img_path_SD)
        img_FP = Image.open(img_path_FP)
        

        FA_markers = ["FA{:02d}".format(i) for i in [2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
        SD_markers = ["SD{:02d}".format(i) for i in [1,2,3,4,5,7,8,9,10,11]]
        FP_markers = ["FP{:02d}".format(i) for i in [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]

        width, height = self.size

        img_FA = self.center_crop_resize(img_FA, (int(0.375 * width), height), FA_markers, item)
        img_SD = self.center_crop_resize(img_SD, (int(0.25 * width), height), SD_markers, item)
        img_FP = self.center_crop_resize(img_FP, (int(0.375 * width), height), FP_markers, item)
        
        image = self.concatenate_three_views(img_FA, img_SD, img_FP)

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