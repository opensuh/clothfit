
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class AEData(Dataset):
    def __init__(self, imgs_dir, img_res, transform=transforms.ToTensor()):
        super().__init__()
        self.imgs_dir = imgs_dir
        self.img_res = img_res
        assert os.path.exists(imgs_dir), 'Directory {} does not exist.'.format(imgs_dir)
        self.imgs_dir = sorted([os.path.join(imgs_dir, f)
                                for f in os.listdir(imgs_dir)
                                    if f.endswith(".jpg") or f.endswith(".png") or f.endswith('.PNG') or f.endswith(".jpeg")])
        self.transform = transform
    
    def __getitem__(self, index):
        img = Image.open(self.imgs_dir[index]).convert('L')
        img = self.transform(img)
        img = img < 0.5

        return img.float()

    def __len__(self):
        return len(self.imgs_dir)
