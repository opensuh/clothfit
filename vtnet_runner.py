
import argparse
import cv2
import json
import numpy as np
import os
import seaborn as sns
import time
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.vt_data_loader import RealData
from VTNet.clothae import ClothEncoder
from VTNet.unet import *
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class VTNetRunner(object):
    def __init__(self, ae_model, unet_model, img_res):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.feature_extractor = ClothEncoder(image_size=img_res, kernel_size=3, n_filters=32)
        self.feature_extractor.load_state_dict(torch.load(ae_model))

        self.unet = UNet(n_channels=1, n_classes=2, img_res=img_res, bilinear=False)
        self.unet.load_state_dict(torch.load(unet_model))

        self.img_res = img_res

    #-------------------------------------------------------------------------------------------------------------------

    def run_model(self, data_loader, save_path):
        self.feature_extractor.to(self.device)
        self.unet.to(self.device)

        self.feature_extractor.eval()
        self.feature_extractor.requires_grad_(False)
        self.unet.eval()
        self.unet.requires_grad_(False)

        start_time = time.time()
        with torch.no_grad():
            
            count = 0
            for input_cloth, cloth_meas, frontview, body_meas in data_loader:
                input_cloth = input_cloth.to(device=self.device)
                features = self.feature_extractor(input_cloth)

                cloth_meas = cloth_meas.to(device=self.device)
                frontview = frontview.to(device=self.device)
                body_meas = body_meas.to(device=self.device)
                unet_result = self.unet(frontview, features, cloth_meas, body_meas)

                for result in unet_result:
                    result = result.detach().to('cpu').permute(1, 2, 0).numpy()

                    # result = np.concatenate((result, np.zeros((result.shape[0], result.shape[1], 1))), axis=-1)
                    result = np.concatenate((np.zeros((self.img_res, self.img_res, 1)), \
                                            np.expand_dims(result[:, :, 1], axis=-1), \
                                            np.expand_dims(result[:, :, 0], axis=-1)), axis=-1)
                    
                    recons_result = np.multiply(result, 255.0)
                    
                    cv2.imwrite(os.path.join(save_path, f'VTNet_result{str(count)}.png'), recons_result)        
                    count += 1

        elapsed_time = time.time() - start_time
        print('VTNet running time={:.2f}s'.format(elapsed_time))

#-----------------------------------------------------------------------------------------------------------------------

class Config:
    def __init__(self):
        pass

#-----------------------------------------------------------------------------------------------------------------------

def parse_configuration(config_file):
    try:
        with open(config_file) as c:
            configuration = json.load(c)
    except:
        print(f'Failed to open configuration file {config_file}.')
        exit()
    
    arguments = configuration['Arguments'][0]
    config = Config()
    config.ae_model = arguments['ae_model']
    config.unet_model = arguments['unet_model']
    config.img_resolution = arguments['img_resolution']
    config.input_cloth_path = arguments['input_cloth_path']
    config.cloth_measurement_path = arguments['cloth_measurement_path']
    config.body_measurement_path = arguments['body_measurement_path']
    config.frontview_path = arguments['frontview_path']
    config.save_path = arguments['save_path']

    return config

#-----------------------------------------------------------------------------------------------------------------------

def validate_configuration(config):
    assert os.path.exists(config.ae_model), f'AE model weight file {config.ae_model} does not exist.'
    assert os.path.exists(config.unet_model), f'UNet model weight file {config.unet_model} does not exist.'
    assert os.path.exists(config.input_cloth_path), f'Input cloth directory {config.input_cloth_path} does not exist.'
    assert os.path.exists(config.cloth_measurement_path), \
                        f'Cloth measurement directory {config.cloth_measurement_path} does not exist.'
    assert os.path.exists(config.body_measurement_path), \
                        f'Body measurement directory {config.body_measurement_path} does not exists.' 
    assert os.path.exists(config.frontview_path), f'Frontview image directory {config.frontview_path} does not exist.'

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

#-----------------------------------------------------------------------------------------------------------------------

def run(args):
    assert os.path.exists(args.config_file), print(f"Configuration file {args.config_file} does not exist.")

    config = parse_configuration(args.config_file)
    validate_configuration(config)

    transform_img = transforms.Compose([
                transforms.Resize((config.img_resolution, config.img_resolution)),
                transforms.ToTensor()
                ])
    data = RealData(img_res=config.img_resolution, \
                    input_cloth_dir=config.input_cloth_path, \
                    measurement_dir=config.cloth_measurement_path, \
                    frontview_imgs_dir=config.frontview_path, \
                    body_measurement_dir=config.body_measurement_path, \
                    transform=transform_img)
    dataloader = DataLoader(data, batch_size=8, shuffle=False)
    model = VTNetRunner(config.ae_model, config.unet_model, config.img_resolution)
    print('Start Virtual Try-on Network')
    model.run_model(dataloader, config.save_path)
    print('Done')

#=======================================================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./VTNet_test/config.json')

    args = parser.parse_args()
    run(args)
