import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class RMSNet(nn.Module):

    def __init__(self, num_class=4):
        super(RMSNet, self).__init__()
        self.feature_extractor = models.resnet152(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *(list(self.feature_extractor.children())[:-1]))

        # freeze the layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.temporal_layers = nn.Sequential(
            nn.Conv1d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1), nn.ReLU(),
            nn.Conv1d(in_channels=256,
                      out_channels=128,
                      kernel_size=3,
                      padding=1), nn.Dropout(p=0.1))
        self.fc3 = nn.Linear(128, 64)
        self.fc_cls = nn.Linear(64, num_class)
        self.fc_reg = nn.Linear(64, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, 3, 224, 398)
        feat = self.feature_extractor(x).view(batch_size, -1, 2048)
        feat = self.fc2(F.relu(self.fc1(feat))).transpose(1, 2)
        feat = self.temporal_layers(feat)
        feat = torch.max(feat, -1)[0]
        feat = self.fc3(feat)
        # print(feat.shape)
        out_cls = self.fc_cls(feat)
        out_reg = torch.sigmoid(self.fc_reg(feat))
        return out_cls, out_reg
