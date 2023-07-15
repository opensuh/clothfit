
import numpy as np 
import torch
import torchvision.models as models
from torchvision import transforms
import argparse
import os
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
import warnings
import time
from VTNet.att_classifier import AttributeClassifier
from utils.ac_data_loader import ACData
from utils.ae_pytorchtools import EarlyStopping

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

class ACTrainer(object):
    def __init__(self, device, gpu, batch_size, img_res, fold):
        self.device = device
        self.gpu = gpu
        self.batch_size = batch_size

        self.losses = {'type' : nn.CrossEntropyLoss(), \
                       'attributes' : nn.L1Loss()}
        
        self.weight_alpha = 10
        self.epochs = None
        self.img_res = img_res
        self.fold = fold

        # define Feature Extractor model
        self.model = models.resnet18(weights=None)
        # self.model = models.resnet50(weights=None)
        # self.model = models.efficientnet_v2_s(weights=False, num_classes=90)
        self.model.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)

        #Call AC
        self.ac = AttributeClassifier(img_res=self.img_res, n_class=4, in_channels=2, n_filters=32, kernel_size=3)

        self.metrics_type = ['Accuracy', 'Macro F1 Score']
        self.metrics_att = ['MAE', 'MSE']

        #tensorboard
        self.writer = SummaryWriter('./logs_ac')

    #-------------------------------------------------------------------------------------------------------------------

    def train_model(self, train_loader, valid_loader, epochs = 50, learning_rate= 1e-4, betas = (0.9, 0.99)):

        # image_size is the size of input mask
        self.epochs = epochs

        print('Reset trainable parameters')
        self.ac.apply(reset_weights)
        self.ac.to(self.device)
        self.model.to(self.device)

        optimizer_model = torch.optim.Adam(self.model.parameters(), lr = learning_rate, betas= betas, weight_decay=1e-6)
        optimizer_ac = torch.optim.Adam(self.ac.parameters(), lr = learning_rate, betas= betas, weight_decay=1e-6)
        # optimizer_ac = torch.optim.Adam(list(self.model.parameters())+list(self.ac.parameters()), lr = learning_rate, betas= betas, weight_decay=1e-4)
        
        reconstruction_criterion = self.losses

        cuda = True if torch.cuda.is_available() else False
        
        if cuda:
            reconstruction_criterion['type'].cuda()
            reconstruction_criterion['attributes'].cuda()

        early_stopping = EarlyStopping(patience=20)
        try:
            os.mkdir("weights")
        except:
            pass
        
        for epoch in range(epochs):
            #train model
            start_time = time.time()
            epoch_loss = 0
            epoch_type_loss = 0
            epoch_att_loss = 0
            
            self.model.train()
            self.model.requires_grad_(True)
            self.ac.train()
            self.ac.requires_grad_(True)
            
            reconstructed_attributes = []
            original_attributes = []
            step=0

            for gt_img, human_meas, cloth_meas in train_loader:
                # Pass if the batch size is 1 to avoid batchnorm1d
                if gt_img.shape[0] == 1:
                    continue

                gt_img = gt_img.to(device=self.device)
                human_meas = human_meas.to(device=self.device)
                features = self.model(gt_img)
                reconstructed_cloth_type, reconstructed_cloth_att = self.ac(features, human_meas)
                
                cloth_meas = cloth_meas.to(device=self.device)
                type_loss = reconstruction_criterion['type'](reconstructed_cloth_type, cloth_meas[:, 0].to(torch.long))
                attribute_loss = reconstruction_criterion['attributes'](reconstructed_cloth_att, cloth_meas[:, 1:])
                total_loss = type_loss + self.weight_alpha*attribute_loss

                epoch_loss += total_loss.item()
                epoch_type_loss += type_loss.item()
                epoch_att_loss += attribute_loss.item()
                    
                optimizer_model.zero_grad()
                optimizer_ac.zero_grad()
                total_loss.backward()
                optimizer_model.step()
                optimizer_ac.step()

                if step == 0:
                    torch_zero = torch.zeros(1, self.img_res, self.img_res).to(self.device)
                    self.writer.add_image('Train/GT image', torch.cat((gt_img[0], torch_zero), dim=0), epoch, dataformats='CHW')
                
                step+=1
            
            epoch_loss = epoch_loss / step
            epoch_type_loss = epoch_type_loss / step
            epoch_att_loss = epoch_att_loss / step

            self.writer.add_scalar("Train/Loss", epoch_loss, epoch)
            self.writer.add_scalar("Train/Type Loss", epoch_type_loss, epoch)
            self.writer.add_scalar("Train/Attribute Loss", epoch_att_loss, epoch)

            #valid model
            step = 0
            metrics_type_result = {metric_fn : 0 for metric_fn in self.metrics_type}
            metrics_att_result = {metric_fn : [0, 0, 0] for metric_fn in self.metrics_att}
            self.model.eval()
            self.model.requires_grad_(False)
            self.ac.eval()
            self.ac.requires_grad_(False)
            vepoch_loss = 0
            vepoch_type_loss = 0
            vepoch_att_loss = 0

            for vgt_img, vhuman_meas, vcloth_meas in valid_loader:
                # Pass if the batch size is 1 to avoid batchnorm1d
                if vgt_img.shape[0] == 1:
                    continue 

                vgt_img = vgt_img.to(device=self.device)
                vhuman_meas = vhuman_meas.to(device=self.device)
                vfeatures = self.model(vgt_img)
                vreconstructed_cloth_type, vreconstructed_cloth_att = self.ac(vfeatures, vhuman_meas)

                vcloth_meas = vcloth_meas.to(device=self.device)
                vtype_loss = reconstruction_criterion['type'](vreconstructed_cloth_type, vcloth_meas[:, 0].to(torch.long))
                vattribute_loss = reconstruction_criterion['attributes'](vreconstructed_cloth_att, vcloth_meas[:, 1:])
                vtotal_loss = vtype_loss + self.weight_alpha*vattribute_loss
                
                vepoch_loss += vtotal_loss.item()
                vepoch_type_loss += vtype_loss.item()
                vepoch_att_loss += vattribute_loss.item()

                for metric_fn in self.metrics_type:
                    if metric_fn == 'Accuracy':
                        value = accuracy_score(vcloth_meas[:, 0].cpu().detach().numpy().flatten(), \
                                                torch.argmax(vreconstructed_cloth_type, dim=1).cpu().detach().numpy().flatten())
                    elif metric_fn == 'Macro F1 Score':
                        value = f1_score(vcloth_meas[:, 0].cpu().detach().numpy().flatten(), \
                                        torch.argmax(vreconstructed_cloth_type, dim=1).cpu().detach().numpy().flatten(), \
                                        average='macro')
                    metrics_type_result[metric_fn] += value

                for metric_fn in self.metrics_att:
                    if metric_fn == 'MAE':
                        value_chest = mean_absolute_error(vcloth_meas[:, 1].cpu().detach().numpy(), \
                                                    vreconstructed_cloth_att[:, 0].cpu().detach().numpy())
                        value_length = mean_absolute_error(vcloth_meas[:, 2].cpu().detach().numpy(), \
                                                    vreconstructed_cloth_att[:, 1].cpu().detach().numpy())
                        value_sleeve = mean_absolute_error(vcloth_meas[:, 3].cpu().detach().numpy(), \
                                                    vreconstructed_cloth_att[:, 2].cpu().detach().numpy())
                    elif metric_fn == 'MSE':
                        value_chest = mean_squared_error(vcloth_meas[:, 1].cpu().detach().numpy(), \
                                                    vreconstructed_cloth_att[:, 0].cpu().detach().numpy())
                        value_length = mean_squared_error(vcloth_meas[:, 2].cpu().detach().numpy(), \
                                                    vreconstructed_cloth_att[:, 1].cpu().detach().numpy())
                        value_sleeve = mean_squared_error(vcloth_meas[:, 3].cpu().detach().numpy(), \
                                                    vreconstructed_cloth_att[:, 2].cpu().detach().numpy())
                    metrics_att_result[metric_fn][0] += value_chest
                    metrics_att_result[metric_fn][1] += value_length
                    metrics_att_result[metric_fn][2] += value_sleeve


                for i in range(vcloth_meas.size(dim=0)):
                    vrecon_cloth_meas = torch.cat((torch.argmax(vreconstructed_cloth_type[i]).reshape(1), \
                                                    vreconstructed_cloth_att[i]), axis=0)
                    vrecon_cloth_meas = vrecon_cloth_meas.detach().to('cpu').numpy()
                    reconstructed_attributes.append(vrecon_cloth_meas)

                    vorig_cloth_meas = vcloth_meas[i]
                    vorig_cloth_meas = vorig_cloth_meas.detach().to('cpu').numpy()
                    original_attributes.append(vorig_cloth_meas)

                step+=1
            
            vepoch_loss = vepoch_loss / step
            vepoch_type_loss = vepoch_type_loss / step
            vepoch_att_loss = vepoch_att_loss / step

            self.writer.add_scalar("Valid/Loss", vepoch_loss, epoch)
            self.writer.add_scalar("Valid/Type Loss", vepoch_type_loss, epoch)
            self.writer.add_scalar("Valid/Attribute Loss", vepoch_att_loss, epoch)

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
            print('Epoch {}/{} Time={:.2f}s train_loss={:.4f} valid_loss={:.4f}'.format(
                                    epoch +1, epochs,
                                    elapsed_time,
                                    epoch_loss,
                                    vepoch_loss))
            print('\t train type loss={:.4f} train attribute loss={:.4f}'.format(
                                epoch_type_loss, epoch_att_loss))
            print('\t valid type loss={:.4f} valid attribute loss={:.4f}'.format(
                                vepoch_type_loss, vepoch_att_loss))
            print('\t Evaluation for valid type: ', metrics_type_result)
            print('\t Evaluation for valid attribute: ', metrics_att_result)

            file = open(f'Attribute_classifier_result{self.fold}.txt', 'w')
            for i in range(len(reconstructed_attributes)):
                file.write(f'Original{i}: {original_attributes[i]}')
                file.write(f'\tReconstructed{i}: {reconstructed_attributes[i]}')
                file.write('\n')
            file.close()
            
            early_stopping(vepoch_loss, \
                            self.model, f'weights/feature_extractor_f{self.fold}.pth', \
                            self.ac, f'weights/attribute_classifier_f{self.fold}.pth')
            if early_stopping.early_stop:
                print('Early stopping')
                break
        
            self.writer.flush()
        self.writer.close()

    #-------------------------------------------------------------------------------------------------------------------

    def test_model(self, test_loader):

        #load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(f'weights/feature_extractor_f{self.fold}.pth'))
        self.ac.load_state_dict(torch.load(f'weights/attribute_classifier_f{self.fold}.pth'))

        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.ac.to(self.device)
        self.ac.eval()
        self.ac.requires_grad_(False)
        
        ac_loss = self.losses

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            ac_loss['type'].cuda()
            ac_loss['attributes'].cuda()       

        start_time = time.time()

        metrics_type_result = {metric_fn : 0 for metric_fn in self.metrics_type}
        metrics_att_result = {metric_fn : [0, 0, 0] for metric_fn in self.metrics_att}
        with torch.no_grad():
            recon_loss = 0
            recon_type_loss = 0
            recon_att_loss = 0

            step = 0
            reconstructed_attributes = []
            original_attributes = []

            for gt_img, human_meas, cloth_meas in test_loader:
                # Pass if the batch size is 1 to avoid batchnorm1d
                if gt_img.shape[0] == 1:
                    continue 

                gt_img = gt_img.to(device=self.device)
                human_meas = human_meas.to(device=self.device)
                features = self.model(gt_img)
                reconstructed_cloth_type, reconstructed_cloth_att = self.ac(features, human_meas)

                cloth_meas = cloth_meas.to(device=self.device)
                type_loss = ac_loss['type'](reconstructed_cloth_type, cloth_meas[:, 0].to(torch.long))
                attribute_loss = ac_loss['attributes'](reconstructed_cloth_att, cloth_meas[:, 1:])
                total_loss = type_loss + self.weight_alpha*attribute_loss
                
                recon_loss += total_loss.item()
                recon_type_loss += type_loss.item()
                recon_att_loss += attribute_loss.item()

                for metric_fn in self.metrics_type:
                    if metric_fn == 'Accuracy':
                        value = accuracy_score(cloth_meas[:, 0].cpu().detach().numpy().flatten(), \
                                                torch.argmax(reconstructed_cloth_type, dim=1).cpu().detach().numpy().flatten())
                    elif metric_fn == 'Macro F1 Score':
                        value = f1_score(cloth_meas[:, 0].cpu().detach().numpy().flatten(), \
                                        torch.argmax(reconstructed_cloth_type, dim=1).cpu().detach().numpy().flatten(), \
                                        average='macro')
                    metrics_type_result[metric_fn] += value
                
                for metric_fn in self.metrics_att:
                    if metric_fn == 'MAE':
                        value_chest = mean_absolute_error(cloth_meas[:, 1].cpu().detach().numpy(), \
                                                    reconstructed_cloth_att[:, 0].cpu().detach().numpy())
                        value_length = mean_absolute_error(cloth_meas[:, 2].cpu().detach().numpy(), \
                                                    reconstructed_cloth_att[:, 1].cpu().detach().numpy())
                        value_sleeve = mean_absolute_error(cloth_meas[:, 3].cpu().detach().numpy(), \
                                                    reconstructed_cloth_att[:, 2].cpu().detach().numpy())
                    elif metric_fn == 'MSE':
                        value_chest = mean_squared_error(cloth_meas[:, 1].cpu().detach().numpy(), \
                                                    reconstructed_cloth_att[:, 0].cpu().detach().numpy())
                        value_length = mean_squared_error(cloth_meas[:, 2].cpu().detach().numpy(), \
                                                    reconstructed_cloth_att[:, 1].cpu().detach().numpy())
                        value_sleeve = mean_squared_error(cloth_meas[:, 3].cpu().detach().numpy(), \
                                                    reconstructed_cloth_att[:, 2].cpu().detach().numpy())
                    metrics_att_result[metric_fn][0] += value_chest
                    metrics_att_result[metric_fn][1] += value_length
                    metrics_att_result[metric_fn][2] += value_sleeve

                step+=1

                for i in range(cloth_meas.size(dim=0)):
                    recon_cloth_meas = torch.cat((torch.argmax(reconstructed_cloth_type[i]).reshape(1), \
                                                    reconstructed_cloth_att[i]), axis=0)
                    recon_cloth_meas = recon_cloth_meas.detach().to('cpu').numpy()
                    reconstructed_attributes.append(recon_cloth_meas)

                    orig_cloth_meas = cloth_meas[i]
                    orig_cloth_meas = orig_cloth_meas.detach().to('cpu').numpy()
                    original_attributes.append(orig_cloth_meas)

            recon_loss = recon_loss / step
            recon_type_loss = recon_type_loss / step
            recon_att_loss = recon_att_loss / step

            for metric_fn in self.metrics_type:
                metrics_type_result[metric_fn] /= step
            for metric_fn in self.metrics_att:
                for i in range(3):
                    metrics_att_result[metric_fn][i] /= step

            elapsed_time = time.time() - start_time
            print('Test Time={:.2f}s test_loss={:.4f}'.format(elapsed_time, recon_loss))
            print('\t Test type loss={:.4f} Test attribute loss={:.4f}'.format(
                                recon_type_loss, recon_att_loss))
            print('\t Evaluation for test type: ', metrics_type_result)
            print('\t Evaluation for test attribute: ', metrics_att_result)
        
        file = open(f'Attribute_classifier_result{self.fold}.txt', 'a')
        for i in range(len(reconstructed_attributes)):
            file.write(f'Original{i}: {original_attributes[i]}')
            file.write(f'\tReconstructed{i}: {reconstructed_attributes[i]}')
            file.write('\n')
        file.close()

#-----------------------------------------------------------------------------------------------------------------------

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

#=======================================================================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='1', help="GPU number")
    parser.add_argument("--gt_imgs_dir", type=str, default='./Total_dataset/gt_only/', help="Path of groundtruth images directory")
    parser.add_argument("--cloth_measurement_dir", type=str, default='./Total_dataset/cloth_att_only/', \
                        help="Path of cloth measurement directory")
    parser.add_argument("--body_measurement_dir", type=str, default='./Total_dataset/human_factor_only/', \
                        help="Path of body measurement directory")
    parser.add_argument("--img_resolution", type=int, default=512, help="Desired image resolution")
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #parameters
    epochs = 300
    learning_rate= 1e-3
    betas = (0.9, 0.99)
    k_folds = 5

    transform_img = transforms.Compose([
                transforms.Resize((args.img_resolution, args.img_resolution)),
                transforms.ToTensor()
                ])

    trainset = ACData(args.img_resolution, args.gt_imgs_dir, args.cloth_measurement_dir, args.body_measurement_dir, transform=transform_img) 

    for fold in range(k_folds):
        print(f'\nFold {fold}')
        print('-----------------------------------')
        testIndex = [i for i in range(len(trainset)) if i % k_folds == fold]
        trainIndex = [i for i in range(len(trainset)) if i not in testIndex]

        trainIndex, validIndex = random_split(trainIndex, [0.9, 0.1])

        train_subset = Subset(trainset, trainIndex)
        valid_subset = Subset(trainset, validIndex)
        test_subset = Subset(trainset, testIndex)

        trainloader = DataLoader(train_subset, shuffle=True, batch_size=args.batch_size, num_workers=8, drop_last=True)
        validloader = DataLoader(valid_subset, shuffle=True, batch_size=args.batch_size, num_workers=8)
        testloader = DataLoader(test_subset, shuffle=True, batch_size=args.batch_size, num_workers=8)

        model = ACTrainer(device=device, \
                        gpu=args.gpu, \
                        batch_size=args.batch_size, \
                        img_res=args.img_resolution, \
                        fold=fold)

        print("Training Attribute Classifier Started")
        model.train_model(trainloader, validloader, epochs=epochs, learning_rate=learning_rate, betas=betas)
        print("Training Done")
        print("Test Attribute Classifier Started")
        model.test_model(testloader)
        print("Test Done")
