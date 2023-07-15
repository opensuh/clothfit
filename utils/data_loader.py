
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class UNetData(Dataset):
    def __init__(self, 
                img_res,
                input_cloth_dir, 
                measurement_dir, 
                frontview_imgs_dir, 
                body_measurement_dir, 
                groundtruth_dir=None,
                transform=transforms.ToTensor(),
                transform_gt=transforms.ToTensor()):
        super().__init__()
        self.img_res = img_res
        self.input_cloth_dir = input_cloth_dir
        self.measurement_dir = measurement_dir
        self.frontview_imgs_dir = frontview_imgs_dir
        self.body_measurement_dir = body_measurement_dir
        self.groundtruth_dir = groundtruth_dir
        self.transform = transform
        self.transform_gt = transform_gt

        assert os.path.exists(input_cloth_dir), 'Directory {} does not exists.'.format(input_cloth_dir)
        assert os.path.exists(measurement_dir), 'Directory {} does not exists.'.format(measurement_dir)
        assert os.path.exists(frontview_imgs_dir), 'Directory {} does not exists.'.format(frontview_imgs_dir)
        assert os.path.exists(body_measurement_dir), 'Directory {} does not exists.'.format(body_measurement_dir)
        if not groundtruth_dir == None:
            assert os.path.exists(groundtruth_dir), 'Directory {} does not exists.'.format(groundtruth_dir)

        self.input_cloth_dir = sorted([os.path.join(input_cloth_dir, f)
                                        for f in os.listdir(input_cloth_dir)
                                            if f.endswith('.jpg') \
                                                or f.endswith('.png') \
                                                or f.endswith('.PNG') \
                                                or f.endswith('jpeg') \
                                                or f.endswith('JPG')])
        self.measurement_dir = sorted([os.path.join(measurement_dir, f)
                                        for f in os.listdir(measurement_dir)
                                            if f.endswith('.npy')])
        self.frontview_imgs_dir = sorted([os.path.join(frontview_imgs_dir, f)
                                        for f in os.listdir(frontview_imgs_dir)
                                            if f.endswith('.jpg') \
                                                or f.endswith('.png') \
                                                or f.endswith('.PNG') \
                                                or f.endswith('jpeg') \
                                                or f.endswith('JPG')])
        self.body_measurement_dir = sorted([os.path.join(body_measurement_dir, f)
                                        for f in os.listdir(body_measurement_dir)
                                            if f.endswith('.npy')])
        if not groundtruth_dir == None:
            self.groundtruth_dir = sorted([os.path.join(groundtruth_dir, f)
                                        for f in os.listdir(groundtruth_dir)
                                            if f.endswith('.jpg') \
                                                or f.endswith('png') \
                                                or f.endswith('.PNG') \
                                                or f.endswith('jpeg') \
                                                or f.endswith('JPG')])

    #-------------------------------------------------------------------------------------------------------------------

    def __getitem__(self, index):
        input_cloth_img = Image.open(self.input_cloth_dir[index]).convert('L')
        input_cloth_img = self.transform(input_cloth_img)
        input_cloth_img = input_cloth_img < 0.5
        
        cloth_measurement = np.load(self.measurement_dir[index])
        cloth_measurement[1:] *= 100

        frontview_img = Image.open(self.frontview_imgs_dir[index]).convert('L')
        frontview_img = self.transform_gt(frontview_img)
        frontview_img = frontview_img < 0.7

        body_measurement = np.load(self.body_measurement_dir[index])
        body_measurement *= 100

        if not self.groundtruth_dir == None:
            gt_img = Image.open(self.groundtruth_dir[index]).convert('RGB')
            gt_img = self.transform_gt(gt_img)[0:2]

            return input_cloth_img.float(), \
                    torch.from_numpy(cloth_measurement).float(), \
                    frontview_img.float(), \
                    torch.from_numpy(body_measurement).float(), \
                    gt_img
        
        return input_cloth_img.float(), \
                    torch.from_numpy(cloth_measurement).float(), \
                    frontview_img.float(), \
                    torch.from_numpy(body_measurement).float()

    #-------------------------------------------------------------------------------------------------------------------

    def __len__(self):
        return len(self.input_cloth_dir)