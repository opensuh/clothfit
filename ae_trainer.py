import numpy as np 
import torch
import argparse
import os
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils.losses import MAASE
import cv2
import warnings
import time
from VTNet.clothae import *
from utils.ae_data_loader import AEData
from utils.ae_pytorchtools import EarlyStopping

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# To evaluate our Auto Encoder, we compare it with PCA method
class PCATrainer(object):
    def __init__(self, num_components, loss, img_res):
        self.pca = PCA(n_components=num_components)

        losses = {'bce' : nn.BCELoss(), \
                  'mse' : nn.MSELoss(),\
                  'mae' : nn.L1Loss(),\
                  'mae+mse': MAASE()}
        
        self.loss = losses[loss]

        self.img_res = img_res
    
    #-------------------------------------------------------------------------------------------------------------------
    
    def fit_model(self, dataloader):
        count = 0
        for data in dataloader:
            data = data.reshape(data.shape[0], -1)
            
            if count == 0:
                total_data = data
            else:
                total_data = torch.cat((total_data, data), dim=0)
            count += 1
                
        pca_weights = self.pca.fit(total_data)
    
    #-------------------------------------------------------------------------------------------------------------------
    
    def test_model(self, testloader):
        
        start_time = time.time()
        
        loss = 0
        step = 0
        accuracy = 0
        reconstructed_images = []
        original_images = []

        for data in testloader:
            data_reshape = data.reshape(data.shape[0], -1)

            data_features = self.pca.transform(data_reshape)
            reconstructed_data = self.pca.inverse_transform(data_features)
            reconstructed_data = reconstructed_data.reshape(data.shape)

            data = torch.tensor(data).float()
            reconstructed_data = torch.tensor(reconstructed_data).float()

            loss += self.loss(data, reconstructed_data).item()

            binaryX_hat = reconstructed_data > 0.5
            accuracy += accuracy_score(data.cpu().detach().numpy().flatten(), \
                                    binaryX_hat.cpu().detach().numpy().flatten())

            step += 1

            for i in range(data.size(dim=0)):
                x_hat = reconstructed_data[i]
                x_hat = x_hat.detach().permute(1, 2, 0).numpy()
                recons_x_hat = np.multiply(x_hat, 255.0)
                reconstructed_images.append(recons_x_hat)

                x = data[i]
                x = x.detach().permute(1, 2, 0).numpy()
                orig_x = np.multiply(x, 255.0)
                original_images.append(orig_x)
        
        loss /= step
        accuracy /= step

        elapsed_time = time.time() - start_time
        print('Test Time={:.2f}s test_loss={:.4f} test_accuracy={}'.format(elapsed_time, loss, accuracy))

        try:
            os.mkdir("reconstructed_imgs_pca")
        except:
            pass
        
        for i in range(len(reconstructed_images)):
            concat_images = cv2.hconcat([original_images[i], reconstructed_images[i]])
            cv2.imwrite(f'reconstructed_imgs_pca/recons{str(i)}.png', concat_images)

#-----------------------------------------------------------------------------------------------------------------------
        
class AETrainer(object):
    def __init__(self, device, gpu, batch_size, loss, img_res, k_folds):
        self.device = device
        self.gpu = gpu
        self.batch_size = batch_size

        losses = {'bce' : nn.BCELoss(), \
                  'mse' : nn.MSELoss(),\
                  'mae' : nn.L1Loss(),\
                  'mae+mse': MAASE()}
        
        self.loss = losses[loss]
        self.epochs = None
        self.img_res = img_res
        self.k_folds = k_folds

        self.feature_extractor = ClothEncoder(image_size= self.img_res, kernel_size=3, n_filters=32)
        self.decoder = ClothDecoder(image_size=self.img_res, kernel_size=3, n_filters=32)

        #tensorboard
        self.writer = SummaryWriter('./logs_ae')

    #-------------------------------------------------------------------------------------------------------------------

    def train_model(self, trainloader, validloader, current_fold, epochs = 50, learning_rate= 1e-4, betas = (0.9, 0.99)):
 
        self.epochs = epochs

        print(f'\nFold {current_fold}')
        print('-----------------------------------')

        print('Reset trainable parameters')
        self.feature_extractor.apply(reset_weights)
        self.decoder.apply(reset_weights)

        self.feature_extractor.to(self.device)
        self.decoder.to(self.device)

        optimizer_extractor = torch.optim.Adam(self.feature_extractor.parameters(), lr = learning_rate, betas= betas)
        optimizer_decoder = torch.optim.Adam(self.decoder.parameters(), lr = learning_rate, betas= betas)

        reconstruction_criterion = self.loss

        cuda = True if torch.cuda.is_available() else False
        
        if cuda:
            reconstruction_criterion.cuda()

        early_stopping = EarlyStopping(patience=20)
        try:
            os.mkdir("weights")
        except:
            pass

        # fold_loss_valid = 0

        epoch_counter = 0
        for epoch in range(epochs):
            #train model
            start_time = time.time()
            epoch_loss = 0
            
            self.feature_extractor.train()
            self.decoder.train()
            self.feature_extractor.requires_grad_(True)
            self.decoder.requires_grad_(True)
            
            step=0
            for X in trainloader:
                X = X.to(device=self.device)
                features = self.feature_extractor(X)
                X_hat = self.decoder(features)

                recon_loss = reconstruction_criterion(X_hat, X)

                epoch_loss += recon_loss.item()
                    
                optimizer_extractor.zero_grad()
                optimizer_decoder.zero_grad()
                recon_loss.backward() 
                optimizer_extractor.step()
                optimizer_decoder.step()
                
                if step == 0:
                    self.writer.add_image('Train/GT image', X[0].detach().to('cpu'), epoch, dataformats='CHW')

                step+=1
            
            epoch_loss = epoch_loss / step

            self.writer.add_scalar("Train/Loss", epoch_loss, epoch)

            #valid model
            step = 0
            self.feature_extractor.eval()
            self.decoder.eval()
            vepoch_loss = 0
            reconstructed_images = []
            original_images = []

            for V in validloader:
                V = V.to(device=self.device)
                vfeatures = self.feature_extractor(V)
                V_hat = self.decoder(vfeatures)

                vrecon_loss = reconstruction_criterion(V_hat, V)
                vepoch_loss += vrecon_loss.item()

                for i in range(V.size(dim=0)):
                    v_hat = V_hat[i]
                    v_hat = v_hat.detach().to('cpu').permute(1, 2, 0).numpy()
                    recons_v_hat = np.multiply(v_hat, 255.0)
                    reconstructed_images.append(recons_v_hat)

                    v = V[i]
                    v = v.detach().to('cpu').permute(1, 2, 0).numpy()
                    orig_v = np.multiply(v, 255.0)
                    original_images.append(orig_v)

                step+=1
            
            vepoch_loss = vepoch_loss / step

            self.writer.add_scalar("Valid/Loss", vepoch_loss, epoch)
            self.writer.add_image('ValidImage/GT image', np.divide(original_images[0], 255.0), epoch, dataformats='HWC')
            self.writer.add_image('ValidImage/Reconstructed image', np.divide(reconstructed_images[0], 255.0), epoch, dataformats='HWC')

            elapsed_time = time.time() - start_time 
            print('Epoch {}/{} Time={:.2f}s train_loss={:.4f} valid_loss={:.4f}'.format(
                                    epoch +1, epochs,
                                    elapsed_time,
                                    epoch_loss,
                                    vepoch_loss))

            try:
                os.mkdir("reconstructed_imgs")
            except:
                pass
        
            for i in range(len(reconstructed_images)):
                concat_images = cv2.hconcat([original_images[i], reconstructed_images[i]])
                cv2.imwrite(f'reconstructed_imgs/recons{str(i)}.png', concat_images)
            
            early_stopping(vepoch_loss, \
                            self.feature_extractor, f'weights/cloth_encoder_f{current_fold}.pth', \
                            self.decoder, f'weights/cloth_decoder_f{current_fold}.pth')
            if early_stopping.early_stop:
                print('Early stopping')
                break

            epoch_counter += 1

            self.writer.flush()
        self.writer.close()

    #-------------------------------------------------------------------------------------------------------------------

    def test_model(self, testloader, current_fold):
        #load the last checkpoint with the best model
        self.feature_extractor.load_state_dict(torch.load(f'weights/cloth_encoder_f{current_fold}.pth'))
        self.decoder.load_state_dict(torch.load(f'weights/cloth_decoder_f{current_fold}.pth'))

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
            accuracy = 0
            step = 0
            reconstructed_images = []
            original_images = []
            for X in testloader:
                X = X.to(device=self.device)
                features = self.feature_extractor(X)
                X_hat = self.decoder(features)

                recon_loss += reconstruction_criterion(X_hat, X).item()
                
                binaryX_hat = X_hat > 0.5
                accuracy += accuracy_score(X.cpu().detach().numpy().flatten(), \
                                        binaryX_hat.cpu().detach().numpy().flatten())

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
            accuracy /= step
            elapsed_time = time.time() - start_time
            print('Test Time={:.2f}s test_loss={:.4f} test_accuracy={}'.format(elapsed_time, recon_loss, accuracy))

        try:
            os.mkdir("reconstructed_imgs")
        except:
            pass
        
        for i in range(len(reconstructed_images)):
            concat_images = cv2.hconcat([original_images[i], reconstructed_images[i]])
            cv2.imwrite(f'reconstructed_imgs/recons{str(i)}.png', concat_images)

#-----------------------------------------------------------------------------------------------------------------------

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

#=======================================================================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='1', help="GPU number")
    parser.add_argument("--imgs_dir", type=str, required=True, help="Path of input images directory")
    parser.add_argument("--img_resolution", type=int, default=512, help="Desired image resolution")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--loss", type=str, default='bce', help='choose one: mae, mse, mae+mse, bce')
    parser.add_argument("--model", type=str, default='ae', help='ae or pca')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #parameters
    epochs = 300
    learning_rate= 1e-4
    betas = (0.9, 0.99)
    k_folds = 5

    transform_training = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((args.img_resolution, args.img_resolution)),
                transforms.RandomAffine(degrees=3, translate=(0.05, 0.1), scale=(0.8, 1.0), fill=255),
                transforms.ToTensor()
                ])
    transform_img = transforms.Compose([
                transforms.Resize((args.img_resolution, args.img_resolution)),
                transforms.ToTensor()
                ])

    trainset = AEData(args.imgs_dir, args.img_resolution, transform_training)
    valid_testset = AEData(args.imgs_dir, args.img_resolution, transform_img)

    for fold in range(k_folds):
        testIndex = [i for i in range(len(trainset)) if i % k_folds == fold]
        trainIndex = [i for i in range(len(trainset)) if i not in testIndex]

        trainIndex, validIndex = random_split(trainIndex, [0.9, 0.1])

        train_subset = Subset(trainset, trainIndex)
        valid_subset = Subset(valid_testset, validIndex)
        test_subset = Subset(valid_testset, testIndex)

        trainloader = DataLoader(train_subset, batch_size=args.batch_size, num_workers=8, shuffle=True)
        validloader = DataLoader(valid_subset, batch_size=args.batch_size, num_workers=8, shuffle=True)
        testloader = DataLoader(test_subset, batch_size=args.batch_size, num_workers=8, shuffle=True)

        if args.model == 'ae':
            model = AETrainer(device=device, \
                        gpu=args.gpu, \
                        batch_size=args.batch_size, \
                        loss = args.loss, \
                        img_res=args.img_resolution, \
                        k_folds=k_folds)

            print("Training AE Started")
            model.train_model(trainloader, validloader, fold, epochs=epochs, learning_rate=learning_rate, betas=betas)
            print("Training Done")
            print("Test AE Started")
            model.test_model(testloader, fold)
            print("Test Done")
        
        elif args.model == 'pca':
            num_components = 256
            model = PCATrainer(num_components=num_components, loss=args.loss, img_res = args.img_resolution)
            print("Training PCA Started")
            model.fit_model(trainloader)
            print("Training Done")
            print("Test PCA Started")
            model.test_model(testloader)
            print("Test Done")
            break
        