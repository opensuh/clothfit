
import math
import torch
import torch.nn as nn

class AttributeClassifier(nn.Module):

    def __init__(self, img_res=512, n_class = 4, in_channels=2, n_filters=32, kernel_size=3):
        super(AttributeClassifier, self).__init__()
        self.img_res = img_res

        self.fc_mask = nn.Sequential(
                nn.Linear(1000, 256),
                nn.BatchNorm1d(256, track_running_stats=False),
                nn.Dropout(0.3),
                nn.ReLU()
        )
        self.fc_human_att = nn.Sequential(
                nn.Linear(2, 44),
                nn.ReLU()
                # nn.Tanh()
        )

        self.fc_type1 = nn.Sequential(
                nn.Linear(256, 128),
                nn.BatchNorm1d(128, track_running_stats=False),
                nn.Dropout(0.3),
                nn.ReLU(),
                
        )
        self.fc_type2 = nn.Sequential(
                nn.Linear(128, n_class)
        )

        self.fc_att1 = nn.Sequential(
                nn.Linear(258, 128),
                nn.ReLU()

        )
        self.fc_att2 = nn.Sequential(
                nn.Linear(128, 3)
        )

        
    def forward(self, clothed_body_features, human_param):
        features = self.fc_mask(clothed_body_features)
        cloth_type = self.fc_type1(features)
        cloth_type = self.fc_type2(cloth_type)

        # human_emb = self.fc_human_att(human_param)

        concat_features = torch.cat((features, human_param), axis = -1)
        concat_features = self.fc_att1(concat_features)
        cloth_att = self.fc_att2(concat_features)

        return cloth_type, cloth_att
