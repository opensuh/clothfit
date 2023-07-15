
import argparse
import cv2
import json
import numpy as np 
import os
import segmentation_models_pytorch as smp
import time
import torch
import torchvision.models as models
import torch.nn as nn
import utils_smp
import warnings

from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils.data_loader import UNetData
from utils.losses import MAASE
from utils.unet_pytorchtools import EarlyStopping
from utils.vt_data_loader import RealData
from VTNet.att_classifier import *
from VTNet.clothae import *
from VTNet.unet import *

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

class UNetTrainer(object):
    def __init__(self, device, ae_model, ac_extractor_model, ac_model, batch_size, loss, n_channels, n_classes, img_res, real_set, bilinear, k_folds, render=True):
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
        
        #Call Attribute Classifier
        self.ac_extractor = models.resnet18(weights=None)
        self.ac_extractor.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.ac_extractor.load_state_dict(torch.load(ac_extractor_model))

        self.ac = AttributeClassifier(img_res=self.img_res)
        self.ac.load_state_dict(torch.load(ac_model))

        #Fixed models
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        self.feature_extractor.requires_grad_(False)
        self.ac_extractor.to(self.device)
        self.ac_extractor.eval()
        self.ac_extractor.requires_grad_(False)
        self.ac.to(self.device)
        self.ac.eval()
        self.ac.requires_grad_(False)

        #Loss weight
        self.alpha = 100
        self.beta = 0.1
        self.gamma = 5

        #tensorboard
        self.writer = SummaryWriter('./logs_unet')

        #real dataset
        self.realset_loader = real_set

    #-------------------------------------------------------------------------------------------------------------------

    def run_agvton(self, input_cloth, cloth_meas, frontview, body_meas):
        input_cloth = input_cloth.to(device=self.device)
        frontview = frontview.to(device=self.device)
        cloth_meas = cloth_meas.to(device=self.device)
        body_meas = body_meas.to(device=self.device)

        features = self.feature_extractor(input_cloth)
        unet_result = self.unet(frontview, features, cloth_meas, body_meas)

        ac_features = self.ac_extractor(unet_result)
        ac_result_type, ac_result_att = self.ac(ac_features, body_meas)

        return unet_result, ac_result_type, ac_result_att

    #-------------------------------------------------------------------------------------------------------------------

    def train_model(self, data_loader, valid_loader, current_fold, epochs=50, learning_rate=1e-4, betas=(0.9, 0.99)):
        
        print('Reset trainable parameters')
        self.unet.apply(reset_weights)
        self.unet.to(self.device) 

        optimizer_unet = torch.optim.Adam(self.unet.parameters(), lr=learning_rate, betas=betas, weight_decay=1e-4)
        
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

        early_stopping = EarlyStopping(patience=20)

        if not os.path.exists('./weights'):
            os.mkdir('./weights')
        if not os.path.exists(f'./unet_results_f{current_fold}'):
            os.mkdir(f'./unet_results_f{current_fold}')

        for epoch in range(epochs):

            start_time = time.time()

            #train model
            self.unet.train()
            self.unet.requires_grad_(True)
            epoch_loss = 0
            epoch_type_loss = 0
            epoch_att_loss = 0
            epoch_unet_loss = 0
            step = 0

            for input_cloth, cloth_meas, frontview, body_meas, gt in data_loader:
                unet_result, ac_result_type, ac_result_att = self.run_agvton(input_cloth, 
                                                                             cloth_meas, 
                                                                             frontview, 
                                                                             body_meas)

                cloth_meas = cloth_meas.to(device=self.device)
                type_loss = ac_type_loss(ac_result_type, cloth_meas[:, 0].to(torch.long))
                attribute_loss = ac_att_loss(ac_result_att, cloth_meas[:, 1:])

                gt = gt.to(device=self.device)
                synthesis_loss = unet_loss(unet_result, gt.to(torch.int64))
                loss = self.alpha * synthesis_loss + self.beta * type_loss + self.gamma * attribute_loss
                
                epoch_loss += loss.item()
                epoch_type_loss += type_loss.item()
                epoch_att_loss += attribute_loss.item()
                epoch_unet_loss += synthesis_loss.item()

                optimizer_unet.zero_grad()
                loss.backward()
                optimizer_unet.step()

                step += 1

            epoch_loss = epoch_loss / step
            epoch_type_loss /= step
            epoch_att_loss /= step
            epoch_unet_loss /= step

            self.writer.add_scalar("Train/Total Loss", epoch_loss, epoch)
            self.writer.add_scalar("Train/Unet Loss", epoch_unet_loss, epoch)
            self.writer.add_scalar("Train/Cloth Type Loss", epoch_type_loss, epoch)
            self.writer.add_scalar("Train/Cloth Attribute Loss", epoch_att_loss, epoch)

            #valid model
            step = 0
            metrics_result = {metric_fn.__name__ : 0 for metric_fn in self.metrics}
            metrics_type_result = {metric_fn : 0 for metric_fn in self.metrics_type}
            metrics_att_result = {metric_fn : [0, 0, 0] for metric_fn in self.metrics_att}
            result_imgs = []
            gt_imgs = []

            self.unet.eval()
            self.unet.requires_grad_(False)
            vepoch_loss = 0
            vepoch_type_loss = 0
            vepoch_att_loss = 0
            vepoch_unet_loss = 0

            for vinput_cloth, vcloth_meas, vfrontview, vbody_meas, vgt in valid_loader:
                vunet_result, vac_result_type, vac_result_att = self.run_agvton(vinput_cloth, 
                                                                                vcloth_meas, 
                                                                                vfrontview, 
                                                                                vbody_meas)

                vcloth_meas = vcloth_meas.to(device=self.device)
                vtype_loss = ac_type_loss(vac_result_type, vcloth_meas[:, 0].to(torch.long))
                vattribute_loss = ac_att_loss(vac_result_att, vcloth_meas[:, 1:])

                vgt = vgt.to(device=self.device)
                vsynthesis_loss = unet_loss(vunet_result, vgt.to(torch.int64))
                vloss = self.alpha * vsynthesis_loss + self.beta * vtype_loss + self.gamma * vattribute_loss
                
                vepoch_loss += vloss.item()
                vepoch_type_loss += vtype_loss.item()
                vepoch_att_loss += vattribute_loss.item()
                vepoch_unet_loss += vsynthesis_loss.item()

                #update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(vunet_result, vgt.to(torch.int64)).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                for k, v in metrics_meters.items():
                    metrics_result[k] += v.mean

                for metric_fn in self.metrics_type:
                    if metric_fn == 'Accuracy':
                        value = accuracy_score(vcloth_meas[:, 0].cpu().detach().numpy().flatten(), \
                                                torch.argmax(vac_result_type, dim=1).cpu().detach().numpy().flatten())
                    elif metric_fn == 'Macro F1 Score':
                        value = f1_score(vcloth_meas[:, 0].cpu().detach().numpy().flatten(), \
                                        torch.argmax(vac_result_type, dim=1).cpu().detach().numpy().flatten(), \
                                        average='macro')
                    metrics_type_result[metric_fn] += float(value)

                for metric_fn in self.metrics_att:
                    if metric_fn == 'MAE':
                        value_chest = mean_absolute_error(vcloth_meas[:, 1].cpu().detach().numpy(), \
                                                    vac_result_att[:, 0].cpu().detach().numpy())
                        value_length = mean_absolute_error(vcloth_meas[:, 2].cpu().detach().numpy(), \
                                                    vac_result_att[:, 1].cpu().detach().numpy())
                        value_sleeve = mean_absolute_error(vcloth_meas[:, 3].cpu().detach().numpy(), \
                                                    vac_result_att[:, 2].cpu().detach().numpy())
                    elif metric_fn == 'MSE':
                        value_chest = mean_squared_error(vcloth_meas[:, 1].cpu().detach().numpy(), \
                                                    vac_result_att[:, 0].cpu().detach().numpy())
                        value_length = mean_squared_error(vcloth_meas[:, 2].cpu().detach().numpy(), \
                                                    vac_result_att[:, 1].cpu().detach().numpy())
                        value_sleeve = mean_squared_error(vcloth_meas[:, 3].cpu().detach().numpy(), \
                                                    vac_result_att[:, 2].cpu().detach().numpy())
                    metrics_att_result[metric_fn][0] += float(value_chest)
                    metrics_att_result[metric_fn][1] += float(value_length)
                    metrics_att_result[metric_fn][2] += float(value_sleeve)

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
            vepoch_type_loss /= step
            vepoch_att_loss /= step
            vepoch_unet_loss /= step

            self.writer.add_scalar("Valid/Total Loss", vepoch_loss, epoch)
            self.writer.add_scalar("Valid/Unet Loss", vepoch_unet_loss, epoch)
            self.writer.add_scalar("Valid/Cloth Type Loss", vepoch_type_loss, epoch)
            self.writer.add_scalar("Valid/Cloth Attribute Loss", vepoch_att_loss, epoch)

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
            for metric_fn in self.metrics_type:
                metrics_type_result[metric_fn] /= step
                self.writer.add_scalar(f'Evaluation Cloth Type/{metric_fn}', metrics_type_result[metric_fn], epoch)
            for metric_fn in self.metrics_att:
                for i in range(3):
                    metrics_att_result[metric_fn][i] /= step
                self.writer.add_scalar(f'Evaluation Cloth Att/{metric_fn} Chest', metrics_att_result[metric_fn][0], epoch)
                self.writer.add_scalar(f'Evaluation Cloth Att/{metric_fn} Length', metrics_att_result[metric_fn][1], epoch)
                self.writer.add_scalar(f'Evaluation Cloth Att/{metric_fn} Sleeve', metrics_att_result[metric_fn][2], epoch)
            

            elapsed_time = time.time() - start_time
            print('Epoch {}/{} Time={:.2f}s VTNet train_loss={:.4f} valid_loss={:.4f}' \
                .format(epoch+1, epochs, elapsed_time, epoch_loss, vepoch_loss))
            print('\t train unet loss={:.4f} train type loss={:.4f} train attribute loss={:.4f}' \
                .format(epoch_unet_loss, epoch_type_loss, epoch_att_loss))
            print('\t valid unet loss={:.4f} valid type loss={:.4f} valid attribute loss={:.4f}'\
                .format(vepoch_unet_loss, vepoch_type_loss, vepoch_att_loss))
            
            print('\t Evaluation for valid set: ', metrics_result)
            print('\t Evaluation for valid cloth type: ', metrics_type_result)
            print('\t Evaluation for valid cloth attribute: ', metrics_att_result)
            
            if self.render == True:
                for i in range(len(result_imgs)):
                    concat_images = cv2.hconcat([gt_imgs[i], result_imgs[i]])
                    cv2.imwrite(f'./unet_results_f{current_fold}/unet_result{i}.png', concat_images)
                del gt_imgs, result_imgs, concat_images

            #Check with real datasets
            self.test_real_dataset(self.realset_loader, current_fold, epoch)

            early_stopping(-early_stop_score, self.unet, f'./weights/unet_f{current_fold}.pth')
            if early_stopping.early_stop:
                print('Early stopping')
                break

            self.writer.flush()
        self.writer.close()

    #-------------------------------------------------------------------------------------------------------------------

    def test_real_dataset(self, realset_loader, current_fold, current_epoch):

        if not os.path.exists(f'./realset_results_f{current_fold}'):
            os.mkdir(f'./realset_results_f{current_fold}')

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
                
                cv2.imwrite(f'./realset_results_f{current_fold}/VTNet_result{str(count)}.png', recons_result)
                count += 1

    #-------------------------------------------------------------------------------------------------------------------

    def test_model(self, test_loader, current_fold):
        #load the last checkpoint with the best model
        self.unet.load_state_dict(torch.load(f'weights/unet_f{current_fold}.pth'))
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
        
        if not os.path.exists(f'./unet_results_f{current_fold}'):
            os.mkdir(f'./unet_results_f{current_fold}')

        metrics_result = {metric_fn.__name__ : 0 for metric_fn in self.metrics}
        metrics_type_result = {metric_fn : 0 for metric_fn in self.metrics_type}
        metrics_att_result = {metric_fn : [0, 0, 0] for metric_fn in self.metrics_att}
        with torch.no_grad():
            step = 0
            total_loss = 0
            total_type_loss = 0
            total_attribute_loss = 0
            total_synthesis_loss = 0
            result_imgs = []
            gt_imgs = []

            for input_cloth, cloth_meas, frontview, body_meas, gt in test_loader:
                unet_result, ac_result_type, ac_result_att = self.run_agvton(input_cloth, 
                                                                             cloth_meas, 
                                                                             frontview, 
                                                                             body_meas)

                cloth_meas = cloth_meas.to(device=self.device)
                type_loss = ac_type_loss(ac_result_type, cloth_meas[:, 0].to(torch.long))
                attribute_loss = ac_att_loss(ac_result_att, cloth_meas[:, 1:])

                gt = gt.to(device=self.device)
                synthesis_loss = unet_loss(unet_result, gt.to(torch.int64))
                sum = self.alpha * synthesis_loss + self.beta * type_loss + self.gamma * attribute_loss
               
                total_loss += sum.item()
                total_type_loss += type_loss.item()
                total_attribute_loss += attribute_loss.item()
                total_synthesis_loss += synthesis_loss.item()

                #update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(unet_result, gt.to(torch.int64)).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                for k, v in metrics_meters.items():
                    metrics_result[k] += v.mean
                del metric_value

                for metric_fn in self.metrics_type:
                    if metric_fn == 'Accuracy':
                        value = accuracy_score(cloth_meas[:, 0].cpu().detach().numpy().flatten(), \
                                                torch.argmax(ac_result_type, dim=1).cpu().detach().numpy().flatten())
                    elif metric_fn == 'Macro F1 Score':
                        value = f1_score(cloth_meas[:, 0].cpu().detach().numpy().flatten(), \
                                        torch.argmax(ac_result_type, dim=1).cpu().detach().numpy().flatten(), \
                                        average='macro')
                    metrics_type_result[metric_fn] += value

                for metric_fn in self.metrics_att:
                    if metric_fn == 'MAE':
                        value_chest = mean_absolute_error(cloth_meas[:, 1].cpu().detach().numpy(), \
                                                    ac_result_att[:, 0].cpu().detach().numpy())
                        value_length = mean_absolute_error(cloth_meas[:, 2].cpu().detach().numpy(), \
                                                    ac_result_att[:, 1].cpu().detach().numpy())
                        value_sleeve = mean_absolute_error(cloth_meas[:, 3].cpu().detach().numpy(), \
                                                    ac_result_att[:, 2].cpu().detach().numpy())
                    elif metric_fn == 'MSE':
                        value_chest = mean_squared_error(cloth_meas[:, 1].cpu().detach().numpy(), \
                                                    ac_result_att[:, 0].cpu().detach().numpy())
                        value_length = mean_squared_error(cloth_meas[:, 2].cpu().detach().numpy(), \
                                                    ac_result_att[:, 1].cpu().detach().numpy())
                        value_sleeve = mean_squared_error(cloth_meas[:, 3].cpu().detach().numpy(), \
                                                    ac_result_att[:, 2].cpu().detach().numpy())
                    metrics_att_result[metric_fn][0] += value_chest
                    metrics_att_result[metric_fn][1] += value_length
                    metrics_att_result[metric_fn][2] += value_sleeve

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
            total_type_loss /= step 
            total_attribute_loss /= step
            total_synthesis_loss /= step

            for metric_fn in self.metrics:
                metrics_result[metric_fn.__name__] /= step
            for metric_fn in self.metrics_type:
                metrics_type_result[metric_fn] /= step
            for metric_fn in self.metrics_att:
                for i in range(3):
                    metrics_att_result[metric_fn][i] /= step

            elapsed_time = time.time() - start_time
            print('Test Time={:.2f}s VTNet test_loss={:.4f}'.format(elapsed_time, total_loss))
            print('\t unet loss={:.4f} type loss={:.4f} attribute loss={:.4f}' \
                .format(total_synthesis_loss, total_type_loss, total_attribute_loss))
            
            print('\t Evaluation for test set: ', metrics_result)
            print('\t Evaluation for test cloth type: ', metrics_type_result)
            print('\t Evaluation for test cloth attribute: ', metrics_att_result)

        if self.render == True:
            for i in range(len(result_imgs)):
                concat_images = cv2.hconcat([gt_imgs[i], result_imgs[i]])
                cv2.imwrite(f'./unet_results_f{current_fold}/unet_result{str(i)}.png', concat_images)

#-----------------------------------------------------------------------------------------------------------------------

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

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
    
    config = Config()
    arguments = configuration['Arguments'][0]
    config.ae_weight = arguments['ae_weight']
    config.ac_extractor_weight = arguments['ac_extractor_weight']
    config.ac_weight = arguments['ac_weight']
    config.input_cloth_dir = arguments['input_cloth_dir']
    config.cloth_measurement_dir = arguments['cloth_measurement_dir']
    config.frontview_imgs_dir = arguments['frontview_imgs_dir']
    config.body_measurement_dir = arguments['body_measurement_dir']
    config.groundtruth_dir = arguments['groundtruth_dir']
    config.img_resolution = arguments['img_resolution']
    config.batch_size = arguments['batch_size']
    config.loss = arguments['loss']
    config.real_cloth_dir = arguments['real_cloth_dir']
    config.real_cloth_measurement_dir = arguments['real_cloth_measurement_dir']
    config.real_frontview_dir = arguments['real_frontview_dir']
    config.real_body_measurement_dir = arguments['real_body_measurement_dir']

    return config

#=======================================================================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./config.json')

    args = parser.parse_args()
    config = parse_configuration(args.config_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #parameters
    epochs = 300
    k_folds = 5
    learning_rate = 1e-3
    betas = (0.9, 0.99)
    render = True

    transform_training = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((config.img_resolution, config.img_resolution)),
                transforms.RandomAffine(degrees=3, translate=(0.05, 0.05), scale=(0.8, 1.0), fill=255),
                transforms.ToTensor()
                ])    
    transform_img = transforms.Compose([
                transforms.Resize((config.img_resolution, config.img_resolution)),
                transforms.ToTensor()
                ])
    
    train_dataset = UNetData(config.img_resolution, \
                    config.input_cloth_dir, \
                    config.cloth_measurement_dir, \
                    config.frontview_imgs_dir, \
                    config.body_measurement_dir, \
                    config.groundtruth_dir,
                    transform_training,
                    transform_img)
    
    valid_testset = UNetData(config.img_resolution, \
                    config.input_cloth_dir, \
                    config.cloth_measurement_dir, \
                    config.frontview_imgs_dir, \
                    config.body_measurement_dir, \
                    config.groundtruth_dir,
                    transform_img,
                    transform_img)

    realset = RealData(config.img_resolution, \
                    config.real_cloth_dir, \
                    config.real_cloth_measurement_dir, \
                    config.real_frontview_dir, \
                    config.real_body_measurement_dir, \
                    transform_img)
    
    realsetloader = DataLoader(realset, batch_size=config.batch_size, shuffle=False)

    for fold in range(k_folds):
        print(f'\nFold {fold}')
        print('-----------------------------------')

        testIndex = [i for i in range(len(train_dataset)) if i % k_folds == fold]
        trainIndex = [i for i in range(len(train_dataset)) if i not in testIndex]
        
        trainIndex, validIndex = random_split(trainIndex, [0.9, 0.1])

        train_subset = Subset(train_dataset, trainIndex)
        valid_subset = Subset(valid_testset, validIndex)
        test_subset = Subset(valid_testset, testIndex)

        trainloader = DataLoader(train_subset, batch_size=config.batch_size, num_workers=8, shuffle=True, drop_last=True)
        validloader = DataLoader(valid_subset, batch_size=config.batch_size, num_workers=8, shuffle=True, drop_last=True)
        testloader = DataLoader(valid_testset, batch_size=config.batch_size, num_workers=8, shuffle=True)

        model = UNetTrainer(device=device, \
                        ae_model=config.ae_weight, \
                        ac_extractor_model=config.ac_extractor_weight, \
                        ac_model=config.ac_weight, \
                        batch_size=config.batch_size, \
                        loss=config.loss, \
                        n_channels=1, \
                        n_classes=2, \
                        img_res=config.img_resolution, \
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
        