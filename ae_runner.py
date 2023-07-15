import numpy as np 
import torch
import argparse
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.losses import MAASE
import cv2
import warnings
from time import sleep
import time
from VTNet.clothae import *
from utils.ae_data_loader import AEData

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

class AERunner(object):
    def __init__(self, device, gpu, encoder_path, decoder_path, loss, img_res):
        self.device = device
        self.gpu = gpu

        self.feature_extractor = ClothEncoder(image_size= img_res, kernel_size=3, n_filters=32, feature_dim=306)
        self.decoder = ClothDecoder(image_size=img_res, kernel_size=3, n_filters=32, feature_dim=306)

        self.feature_extractor.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))

        losses = {'bce' : nn.BCELoss(), \
                  'mse' : nn.MSELoss(),\
                  'mae' : nn.L1Loss(),\
                  'mae+mse': MAASE()}
        
        self.loss = losses[loss]

    #-------------------------------------------------------------------------------------------------------------------

    def run_model(self, data_loader):

        self.feature_extractor.to(self.device)
        self.decoder.to(self.device)

        reconstruction_criterion = self.loss

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            reconstruction_criterion.cuda()
        
        start_time = time.time()

        self.feature_extractor.eval()
        self.decoder.eval()

        with torch.no_grad():
            recon_loss = 0
            step = 0
            reconstructed_images = []
            original_images = []
            for X in data_loader:
                X = X.to(device=self.device)
                features = self.feature_extractor(X)
                X_hat = self.decoder(features)

                recon_loss += reconstruction_criterion(X_hat, X).item()
                
                step+=1

                for i in range(X.size(dim=0)):
                    x_hat = X_hat[i]
                    x_hat = x_hat.detach().to('cpu').permute(1, 2, 0).numpy()
                    recons_x_hat = np.multiply(x_hat, 255.0)
                    reconstructed_images.append(recons_x_hat)

                    x = X[i]
                    x = x.detach().to('cpu').permute(1, 2, 0).numpy()
                    orig_x = np.multiply(x, 255.0)
                    original_images.append(orig_x)   

            recon_loss = recon_loss / step
            elapsed_time = time.time() - start_time
            print('Test Time={:.2f}s test_loss={:.4f}'.format(elapsed_time, recon_loss))

        try:
            os.mkdir("ae_result")
        except:
            pass
        
        for i in range(len(reconstructed_images)):
            concat_images = cv2.hconcat([original_images[i], reconstructed_images[i]])
            cv2.imwrite(f'ae_result/recons{str(i)}.png', concat_images)   

#=======================================================================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='1', help="GPU number")
    parser.add_argument("--imgs_dir", type=str, required=True, help="Path of input images directory")
    parser.add_argument("--img_resolution", type=int, default=512, help="Desired image resolution")
    parser.add_argument("--encoder", type=str, default='./weights/cloth_encoder.pth', help="Path of the cloth encoder weights")
    parser.add_argument("--decoder", type=str, default='./weights/cloth_decoder.pth', help="Path of the cloth decoder weights")
    parser.add_argument("--loss", type=str, default='bce', help='choose one: mae, mse, mae+mse, bce')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataset = AEData(args.imgs_dir, args.img_resolution)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = AERunner(device=device, \
                    gpu=args.gpu, \
                    encoder_path=args.encoder, \
                    decoder_path=args.decoder, \
                    loss=args.loss, \
                    img_res=args.img_resolution)
    print("Run AE")
    model.run_model(dataloader)
    print("Done")
