import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernel_size = 3, dropout=False, drop_prob=0.5):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=1)
        )
        
        self.conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size,kernel_size), padding=(kernel_size//2,kernel_size//2)),
                nn.BatchNorm2d(out_channels, track_running_stats=False),
                nn.LeakyReLU(0.1),
        )

    def forward(self, x1):

        up = self.up(x1)
        out = self.conv(up) 
        
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1,1)),
                nn.Sigmoid()
        ) 

    def forward(self, x):
        return self.conv(x)


class MergeFeatures(nn.Module):
    def __init__(self, img_res):
        super(MergeFeatures, self).__init__()
        self.down1 = nn.Conv2d(1024, 32, kernel_size=(3,3), padding=(1,1))
        self.down2 = nn.Conv2d(32, 8, kernel_size=(3,3), padding=(1,1))
        self.pool = nn.MaxPool2d((2,2),(2,2))

        self.current_res = int(img_res / (2**6))
        self.dense = nn.Linear(self.current_res*self.current_res*8, 200)
        # self.dense = nn.Linear(32, 256)

        self.up1 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(8, 32, kernel_size=(1,1), stride=1),
        nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1)))

        self.up2 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(32, 1024, kernel_size=(1,1), stride=1))

        self.res = int(img_res / (2**4))
        self.fc_feature = nn.Sequential(
            nn.Linear(6, 256*self.res*self.res),
            nn.BatchNorm1d(256*self.res*self.res, track_running_stats=False)
        )
    
    def forward(self, x, feature, cloth_param, human_param):
        # x : W/16 x H/16 x 256 , feature: same, cloth_param: 4x1, human_param: 2x1
        attribute = torch.cat ((cloth_param, human_param), 1)
        attribute_emb = self.fc_feature(attribute)
        attribute_emb = attribute_emb.view(-1, 256, self.res, self.res)

        concat_features = torch.cat ((x, feature, attribute_emb), 1)
        
        return concat_features

if __name__ == "__main__":
    mergemerge = MergeFeatures(img_res=128)
    test_x = torch.randn(32,256,8,8)
    test_AE = torch.randn(32,256,8,8)
    test_cloth = torch.randn(32,4)
    test_human = torch.randn(32,2)
    check = mergemerge(test_x, test_AE, test_cloth, test_human)
    print(check.size())