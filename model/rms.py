import torch
import torch.nn as nn
from torchvision import models


class RMSNet(nn.Module):

    def __init__(self, backbone='resnet18', num_class=4):
        super(RMSNet, self).__init__()
        assert backbone in ['resnet18', 'resnet50', 'resnet152']
        if backbone == 'resnet18':
            self.feature_extractor = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT)
            self.feature_extractor = nn.Sequential(
                *(list(self.feature_extractor.children())[:-1]))
            self.neck = nn.Identity()

        elif backbone == 'resnet50':
            self.feature_extractor = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT)
            self.feature_extractor = nn.Sequential(
                *(list(self.feature_extractor.children())[:-1]))
            self.neck = nn.Sequential(nn.Linear(2048, 512), nn.ReLU())

        elif backbone == 'resnet152':
            self.feature_extractor = models.resnet152(
                weights=models.ResNet152_Weights.DEFAULT)
            self.feature_extractor = nn.Sequential(
                *(list(self.feature_extractor.children())[:-1]))
            self.neck = nn.Sequential(nn.Linear(2048, 512), nn.ReLU())

        # freeze the layers
        for i, child in enumerate(self.feature_extractor.children()):
            if i < 7:
                for param in child.parameters():
                    param.requires_grad = False
            elif i == 7:
                for block in child[:-1]:
                    for param in block.parameters():
                        param.requires_grad = False

        self.fc1 = nn.Linear(512, 256)
        self.temporal_layers = nn.Sequential(
            nn.Conv1d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1), nn.ReLU(),
            nn.Conv1d(in_channels=256,
                      out_channels=128,
                      kernel_size=3,
                      padding=1), nn.Dropout(p=0.1))
        self.fc2 = nn.Linear(128, 64)
        self.fc_cls = nn.Linear(64, num_class)
        self.fc_reg = nn.Linear(64, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, 3, 224, 398)
        feat = self.feature_extractor(x).view(batch_size, -1, 2048)
        feat = self.fc1(self.neck(feat)).transpose(1, 2)
        feat = self.temporal_layers(feat)
        feat = torch.max(feat, -1)[0]
        feat = self.fc2(feat)
        # print(feat.shape)
        out_cls = self.fc_cls(feat)
        out_reg = torch.sigmoid(self.fc_reg(feat))
        return out_cls, out_reg
