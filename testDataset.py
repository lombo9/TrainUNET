import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
#import numpy as np



class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if '_segmentation.png' not in f]
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        mask_name = os.path.join(self.mask_dir, self.image_files[idx].replace('.jpg', '_segmentation.png'))
        
        image = Image.open(img_name)
        mask = Image.open(mask_name)  

        #mask_tensor = transforms.ToTensor()(mask)
        #mask_bin = (mask_tensor > 0.3).float()
        #mask = transforms.ToPILImage()(mask_bin)

        if self.transform:
            image = self.transform(image) 
            mask = self.transform(mask)
        return image, mask


def get_isic_dataloader(image_dir, mask_dir, batch_size=32, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    dataset = ISICDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


#if __name__ == '__main__':
    #root_dir = r'C:\Users\lombo\Desktop\3710_report\ISIC-2017_Training_Data\ISIC-2017_Training_Data'
    #dataloader = get_isic_dataloader(root_dir)
    
