
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ACData(Dataset):
    def __init__(self, 
                img_res, 
                groundtruth_img_dir, 
                cloth_measurement_dir,
                body_measurement_dir,
                transform=transforms.ToTensor()):
        super().__init__()
        self.img_res = img_res
        self.groundtruth_img_dir = groundtruth_img_dir
        self.cloth_measurement_dir = cloth_measurement_dir
        self.body_measurement_dir = body_measurement_dir
        self.transform = transform

        assert os.path.exists(groundtruth_img_dir), 'Directory {} does not exists.'.format(groundtruth_img_dir)
        assert os.path.exists(cloth_measurement_dir), 'Directory {} does not exists.'.format(cloth_measurement_dir)
        assert os.path.exists(body_measurement_dir), 'Directory {} does not exists.'.format(body_measurement_dir)

        self.groundtruth_img_dir = sorted([os.path.join(groundtruth_img_dir, f)
                                        for f in os.listdir(groundtruth_img_dir)
                                            if f.endswith('.jpg') \
                                                or f.endswith('.png') \
                                                or f.endswith('.PNG') \
                                                or f.endswith('jpeg') \
                                                or f.endswith('JPG')])
        self.cloth_measurement_dir = sorted([os.path.join(cloth_measurement_dir, f)
                                        for f in os.listdir(cloth_measurement_dir)
                                            if f.endswith('.npy')])
        self.body_measurement_dir = sorted([os.path.join(body_measurement_dir, f)
                                        for f in os.listdir(body_measurement_dir)
                                            if f.endswith('.npy')])

    #-------------------------------------------------------------------------------------------------------------------

    def __getitem__(self, index):
        cloth_measurement = np.load(self.cloth_measurement_dir[index])
        cloth_measurement[1:] *= 100
        
        body_measurement = np.load(self.body_measurement_dir[index])
        body_measurement = body_measurement * 100
                        
        gt_img = Image.open(self.groundtruth_img_dir[index]).convert('RGB')
        gt_img = self.transform(gt_img)[0:2]
         
        return gt_img, \
                torch.from_numpy(body_measurement).float(), \
                torch.from_numpy(cloth_measurement).float()

    #-------------------------------------------------------------------------------------------------------------------

    def __len__(self):
        return len(self.groundtruth_img_dir)