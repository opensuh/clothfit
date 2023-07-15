
import numpy as np 
import torch
import argparse
import os
import segmentation_models_pytorch as smp
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision.models as models
from utils.losses import MAASE, DiceLoss
import utils_smp
import cv2
import warnings
from time import sleep
import time
from VTNet.att_classifier import *
from VTNet.clothae import *
from VTNet.unet import *
from utils.data_loader import Data as unetData
from utils.vt_data_loader import Data as realData
from utils.unet_pytorchtools import EarlyStopping

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

class UNetTrainer(object):
    def __init__(self, device, ae_model, batch_size, loss, n_channels, n_classes, img_res, real_set, bilinear, k_folds, render=True):
        self.device = device
        self.batch_size = batch_size

        self.losses = {'dice': smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, log_loss=True, smooth=0.01), \
                      'focal': smp.losses.FocalLoss(smp.losses.MULTILABEL_MODE), \
                      'ce' : nn.CrossEntropyLoss(), \
                      'mse' : nn.MSELoss(),\
                      'mae' : nn.L1Loss(),\
                      'mae+mse': MAASE()}
        
        self.loss = self.losses[loss]
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.img_res = img_res
        self.bilinear = bilinear
        self.epochs = None
        self.k_folds = k_folds
        self.render = render

        self.metrics = [utils_smp.metrics.IoU(threshold=0.5), utils_smp.metrics.Fscore()]
        self.metrics_type = ['Accuracy', 'Macro F1 Score']
        self.metrics_att = ['MAE', 'MSE']

        #Call Autoencoder for input cloth
        self.feature_extractor = ClothEncoder(image_size=img_res, kernel_size=3, n_filters=32)
        self.feature_extractor.load_state_dict(torch.load(ae_model))

        #Call UNet
        self.unet = UNet(n_channels=self.n_channels, n_classes=self.n_classes, img_res=self.img_res, bilinear=self.bilinear)

        #Fixed models
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        self.feature_extractor.requires_grad_(False)

        #Loss weight
        self.alpha = 1

        #tensorboard
        self.writer = SummaryWriter('./logs_unet_woAC')

        #real dataset
        self.realset_loader = real_set

    #-------------------------------------------------------------------------------------------------------------------

    def train_model(self, data_loader, valid_loader, current_fold, epochs=50, learning_rate=1e-4, betas=(0.9, 0.99)):
        
        self.epochs = epochs
        
        print('Reset trainable parameters')
        self.unet.apply(reset_weights)
        self.unet.to(self.device) 

        optimizer_unet = torch.optim.Adam(self.unet.parameters(), lr=learning_rate, betas=betas, weight_decay=1e-4)
        
        unet_loss = self.loss

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            unet_loss.cuda()
        for metric in self.metrics:
            metric.to(self.device)
        metrics_meters = {metric.__name__: utils_smp.meter.AverageValueMeter() for metric in self.metrics}

        early_stopping = EarlyStopping(patience=20)

        if not os.path.exists('./weights_woAC'):
            os.mkdir('./weights_woAC')
        if not os.path.exists(f'./unet_results_f{current_fold}_woAC'):
            os.mkdir(f'./unet_results_f{current_fold}_woAC')

        for epoch in range(epochs):

            start_time = time.time()

            #train model
            self.unet.train()
            self.unet.requires_grad_(True)
            epoch_loss = 0
            step = 0

            for input_cloth, cloth_meas, frontview, body_meas, gt in data_loader:

                #Extract features of input cloth image using Autoencoder
                input_cloth = input_cloth.to(device=self.device)
                features = self.feature_extractor(input_cloth)

                frontview = frontview.to(device=self.device)
                cloth_meas = cloth_meas.to(device=self.device)
                body_meas = body_meas.to(device=self.device)
                unet_result = self.unet(frontview, features, cloth_meas, body_meas)
                gt = gt.to(device=self.device)

                loss = unet_loss(unet_result, gt.to(torch.int64))
                
                epoch_loss += loss.item()

                optimizer_unet.zero_grad()
                loss.backward()
                optimizer_unet.step()

                step += 1

            epoch_loss = epoch_loss / step

            self.writer.add_scalar("Train/Total Loss", epoch_loss, epoch)

            #valid model
            step = 0
            metrics_result = {metric_fn.__name__ : 0 for metric_fn in self.metrics}
            result_imgs = []
            gt_imgs = []
            self.unet.eval()
            self.unet.requires_grad_(False)
            vepoch_loss = 0

            for vinput_cloth, vcloth_meas, vfrontview, vbody_meas, vgt in valid_loader:
                vinput_cloth = vinput_cloth.to(device=self.device)
                vfeatures = self.feature_extractor(vinput_cloth)

                vcloth_meas = vcloth_meas.to(device=self.device)
                vfrontview = vfrontview.to(device=self.device)
                vbody_meas = vbody_meas.to(device=self.device)
                vunet_result = self.unet(vfrontview, vfeatures, vcloth_meas, vbody_meas)

                vgt = vgt.to(device=self.device)
                vloss = unet_loss(vunet_result, vgt.to(torch.int64))
                
                vepoch_loss += vloss.item()

                #update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(vunet_result, vgt.to(torch.int64)).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                for k, v in metrics_meters.items():
                    metrics_result[k] += v.mean

                if self.render == True:
                    for i in range(vgt.size(dim=0)):
                        result = vunet_result[i]
                        result = result.detach().to('cpu').permute(1, 2, 0).numpy()
                        result = np.concatenate((np.zeros((result.shape[0], result.shape[1], 1)), \
                                                np.expand_dims(result[:, :, 1], axis=-1), \
                                                np.expand_dims(result[:, :, 0], axis=-1)), axis=-1)
                        result = np.multiply(result, 255.0)
                        result_imgs.append(result)

                        vgt_ = vgt[i]
                        vgt_ = vgt_.detach().to('cpu').permute(1, 2, 0).numpy()
                        vgt_ = np.concatenate((np.zeros((vgt_.shape[0], vgt_.shape[1], 1)), \
                                                np.expand_dims(vgt_[:, :, 1], axis=-1), \
                                                np.expand_dims(vgt_[:, :, 0], axis=-1)), axis=-1)
                        vgt_ = np.multiply(vgt_, 255.0)
                        gt_imgs.append(vgt_)

                step += 1
            
            vepoch_loss = vepoch_loss / step

            self.writer.add_scalar("Valid/Total Loss", vepoch_loss, epoch)

            torch_zero = torch.zeros(1, self.img_res, self.img_res).to(self.device)
            self.writer.add_image('ValidImage/GT image', torch.cat((vgt[0, :, :, :], torch_zero), dim=0), epoch)
            
            vunet_result_tensorboard = torch.clamp(vunet_result, 0, 1)
            self.writer.add_image('ValidImage/Result image', torch.cat((vunet_result_tensorboard[0, :, :, :], torch_zero), dim=0), epoch)
            del vunet_result_tensorboard, torch_zero

            early_stop_score = 0
            for metric_fn in self.metrics:
                metrics_result[metric_fn.__name__] /= step
                early_stop_score += metrics_result[metric_fn.__name__]
                self.writer.add_scalar(f'Evaluation Unet/{metric_fn.__name__}', metrics_result[metric_fn.__name__], epoch)

            elapsed_time = time.time() - start_time
            print('Epoch {}/{} Time={:.2f}s VTNet train_loss={:.4f} valid_loss={:.4f}' \
                .format(epoch+1, epochs, elapsed_time, epoch_loss, vepoch_loss))
            print('\t Evaluation for valid set: ', metrics_result)
            
            if self.render == True:
                for i in range(len(result_imgs)):
                    concat_images = cv2.hconcat([gt_imgs[i], result_imgs[i]])
                    cv2.imwrite(f'./unet_results_f{current_fold}_woAC/unet_result{i}.png', concat_images)
                del gt_imgs, result_imgs, concat_images

            #Check with real datasets
            self.test_real_dataset(self.realset_loader, current_fold, epoch)

            early_stopping(-early_stop_score, self.unet, f'./weights_woAC/unet_f{current_fold}.pth')
            if early_stopping.early_stop:
                print('Early stopping')
                break

            self.writer.flush()
        self.writer.close()

    #-------------------------------------------------------------------------------------------------------------------

    def test_real_dataset(self, realset_loader, current_fold, current_epoch):

        if not os.path.exists(f'./realset_results_f{current_fold}_woAC'):
            os.mkdir(f'./realset_results_f{current_fold}_woAC')

        count = 0
        for real_cloth, real_cloth_meas, real_frontview, real_body_meas in realset_loader:
            real_cloth = real_cloth.to(device=self.device)
            real_features = self.feature_extractor(real_cloth)

            real_cloth_meas = real_cloth_meas.to(device=self.device)
            real_frontview = real_frontview.to(device=self.device)
            real_body_meas = real_body_meas.to(device=self.device)
            unet_result = self.unet(real_frontview, real_features, real_cloth_meas, real_body_meas)

            for result in unet_result:
                torch_zero = torch.zeros(1, self.img_res, self.img_res).to(self.device)
                result_tensorboard = torch.clamp(result, 0, 1)
                self.writer.add_image(f'RealImage/img{count}', \
                                        torch.cat((result_tensorboard, torch_zero), dim=0), \
                                        current_epoch, \
                                        dataformats='CHW')

                result = result.detach().to('cpu').permute(1, 2, 0).numpy()
                result = np.concatenate((np.zeros((self.img_res, self.img_res, 1)), \
                                        np.expand_dims(result[:, :, 1], axis=-1), \
                                        np.expand_dims(result[:, :, 0], axis=-1)), axis=-1)
                recons_result = np.multiply(result, 255.0)
                
                cv2.imwrite(f'./realset_results_f{current_fold}_woAC/VTNet_result{str(count)}.png', recons_result)
                count += 1

    #-------------------------------------------------------------------------------------------------------------------

    def test_model(self, test_loader, current_fold):
        #load the last checkpoint with the best model
        self.unet.load_state_dict(torch.load(f'weights_woAC/unet_f{current_fold}.pth'))
        self.unet.to(self.device)
        self.unet.eval()
        self.unet.requires_grad_(False)

        unet_loss = self.loss
        ac_type_loss = self.losses['ce']
        ac_att_loss = self.losses['mae']
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            unet_loss.cuda()
            ac_type_loss.cuda()
            ac_att_loss.cuda()
        for metric in self.metrics:
            metric.to(self.device)
        metrics_meters = {metric.__name__: utils_smp.meter.AverageValueMeter() for metric in self.metrics}
        
        start_time = time.time()
        
        if not os.path.exists(f'./unet_results_f{current_fold}_woAC'):
            os.mkdir(f'./unet_results_f{current_fold}_woAC')

        metrics_result = {metric_fn.__name__ : 0 for metric_fn in self.metrics}
        with torch.no_grad():
            step = 0
            total_loss = 0
            result_imgs = []
            gt_imgs = []

            for input_cloth, cloth_meas, frontview, body_meas, gt in test_loader:
                input_cloth = input_cloth.to(device=self.device)
                features = self.feature_extractor(input_cloth)

                cloth_meas = cloth_meas.to(device=self.device)
                frontview = frontview.to(device=self.device)
                body_meas = body_meas.to(device=self.device)
                unet_result = self.unet(frontview, features, cloth_meas, body_meas)

                gt = gt.to(device=self.device)
                loss = unet_loss(unet_result, gt.to(torch.int64))
               
                total_loss += loss.item()

                #update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(unet_result, gt.to(torch.int64)).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                for k, v in metrics_meters.items():
                    metrics_result[k] += v.mean
                del metric_value

                if self.render == True:
                    for i in range(gt.size(dim=0)):
                        result = unet_result[i]
                        result = result.detach().to('cpu').permute(1, 2, 0).numpy()
                        result = np.concatenate((np.zeros((result.shape[0], result.shape[1], 1)), \
                                                np.expand_dims(result[:, :, 1], axis=-1), \
                                                np.expand_dims(result[:, :, 0], axis=-1)), axis=-1)
                        recons_result = np.multiply(result, 255.0)
                        result_imgs.append(recons_result)

                        gt_ = gt[i]
                        gt_ = gt_.detach().to('cpu').permute(1, 2, 0).numpy()
                        gt_ = np.concatenate((np.zeros((gt_.shape[0], gt_.shape[1], 1)), \
                                            np.expand_dims(gt_[:, :, 1], axis=-1), \
                                            np.expand_dims(gt_[:, :, 0], axis=-1)), axis=-1)
                        orig_gt_ = np.multiply(gt_, 255.0)
                        gt_imgs.append(orig_gt_)
                        
                step += 1
            
            total_loss /= step

            for metric_fn in self.metrics:
                metrics_result[metric_fn.__name__] /= step

            elapsed_time = time.time() - start_time
            print('Test Time={:.2f}s VTNet test_loss={:.4f}'.format(elapsed_time, total_loss))
            print('\t Evaluation for test set: ', metrics_result)

        if self.render == True:
            for i in range(len(result_imgs)):
                concat_images = cv2.hconcat([gt_imgs[i], result_imgs[i]])
                cv2.imwrite(f'./unet_results_f{current_fold}_woAC/unet_result{str(i)}.png', concat_images)

#-----------------------------------------------------------------------------------------------------------------------

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

#=======================================================================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='1', help='GPU number')
    parser.add_argument("--ae_model", type=str, default='./weights/cloth_encoder.pth', \
                        help="File path of Autoencoder model weights")
    parser.add_argument("--input_cloth_dir", type=str, required=True, \
                        help="Path of input cloth images directory")
    parser.add_argument("--cloth_measurement_dir", type=str, required=True, \
                        help="Path of cloth measurement directory")
    parser.add_argument("--frontview_imgs_dir", type=str, required=True, \
                        help="Path of front view images directory")
    parser.add_argument("--body_measurement_dir", type=str, required=True, \
                        help="Path of body measurement directory")
    parser.add_argument("--groundtruth_dir", type=str, required=True, \
                        help="Pasth of ground truth images directory")
    parser.add_argument("--img_resolution", type=int, default=512, help="Desired image resolution")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--loss", type=str, default='dice', \
                        help="Choose one: mae, mse, mae+mse, ce, dice, focal")
    parser.add_argument("--render_result", type=str, default='yes', \
                        help="Render final images of UNet(yes/no)")
    parser.add_argument("--real_cloth_dir", type=str, default='./VTNet_test/input_cloth')
    parser.add_argument("--real_cloth_measurement_dir", type=str, default='./VTNet_test/cloth_measurement')
    parser.add_argument("--real_frontview_dir", type=str, default='./VTNet_test/frontview')
    parser.add_argument("--real_body_measurement_dir", type=str, default='./VTNet_test/body_measurement')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #parameters
    epochs = 300
    k_folds = 5
    learning_rate = 1e-4
    betas = (0.9, 0.99)
    render = True if args.render_result == 'yes' else False

    transform_training = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                # transforms.RandomResizedCrop(size=(args.img_resolution, args.img_resolution), 
                #                             scale=(0.9, 1.2),
                #                             ratio=(1.0, 1.0)),
                transforms.Resize((args.img_resolution, args.img_resolution)),
                transforms.RandomAffine(degrees=3, translate=(0.05, 0.05), scale=(0.8, 1.0), fill=255),
                transforms.ToTensor()
                ])    
    transform_img = transforms.Compose([
                transforms.Resize((args.img_resolution, args.img_resolution)),
                transforms.ToTensor()
                ])
    train_dataset = unetData(args.img_resolution, \
                    args.input_cloth_dir, \
                    args.cloth_measurement_dir, \
                    args.frontview_imgs_dir, \
                    args.body_measurement_dir, \
                    args.groundtruth_dir,
                    transform_training,
                    transform_img)
    
    valid_testset = unetData(args.img_resolution, \
                    args.input_cloth_dir, \
                    args.cloth_measurement_dir, \
                    args.frontview_imgs_dir, \
                    args.body_measurement_dir, \
                    args.groundtruth_dir,
                    transform_img,
                    transform_img)

    realset = realData(args.img_resolution, \
                    args.real_cloth_dir, \
                    args.real_cloth_measurement_dir, \
                    args.real_frontview_dir, \
                    args.real_body_measurement_dir, \
                    transform_img)
    
    realsetloader = DataLoader(realset, batch_size=args.batch_size, shuffle=False)

    for fold in range(k_folds):
        print(f'\nFold {fold}')
        print('-----------------------------------')

        testIndex = [i for i in range(len(train_dataset)) if i % k_folds == fold]
        trainIndex = [i for i in range(len(train_dataset)) if i not in testIndex]
        
        trainIndex, validIndex = random_split(trainIndex, [0.9, 0.1])

        train_subset = Subset(train_dataset, trainIndex)
        valid_subset = Subset(valid_testset, validIndex)
        test_subset = Subset(valid_testset, testIndex)

        trainloader = DataLoader(train_subset, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True)
        validloader = DataLoader(valid_subset, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True)
        testloader = DataLoader(test_subset, batch_size=args.batch_size, num_workers=8, shuffle=True)

        model = UNetTrainer(device=device, \
                        ae_model=args.ae_model, \
                        batch_size=args.batch_size, \
                        loss=args.loss, \
                        n_channels=1, \
                        n_classes=2, \
                        img_res=args.img_resolution, \
                        real_set=realsetloader, \
                        bilinear=False, \
                        k_folds=k_folds, \
                        render=render)
        
        print("Training UNet Started")
        model.train_model(trainloader, validloader, fold, epochs=epochs, learning_rate=learning_rate, betas=betas)
        print("Training Done")
        print("Test UNet Started")
        model.test_model(testloader, fold)
        print("Test Done") 