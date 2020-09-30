""" simple models for model_global """

import torch
import torch.nn as nn
import pdb
import torch.nn.init as init


class Conv1D_Net(nn.Module):
    def __init__(self, inplane, num_classes=1):
        super(Conv1D_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(inplane, inplane, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(inplane),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(inplane, inplane, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(inplane),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(inplane, inplane, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(inplane),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(inplane * 64 , num_classes)
        )
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.softmax(out)
        return out


# class Conv2D_Net(nn.Module):
#     def __init__(self, inplane, num_classes=1):
#         super(Conv2D_Net, self).__init__()
#         self.layer1 = nn.Conv2d(1, 256, kernel_size=(5, 25088), stride=1, padding=(2, 0))
#         self.layer2 = nn.Conv2d(256, 128, kernel_size=(5, 1), stride=1, padding=(2, 0))
#         self.layer3 = nn.Conv2d(128, 64, kernel_size=(5, 1), stride=1, padding=(2, 0))
#         self.layer4 = nn.Conv2d(64, 32, kernel_size=(5, 1), stride=1, padding=(2, 0))
#         self.layer5 = nn.Conv2d(32, 16, kernel_size=(5, 1), stride=1, padding=(2, 0))
#         self.layer6 = nn.Conv2d(16, 8, kernel_size=(5, 1), stride=1, padding=(2, 0))
#         self.relu = nn.ReLU()
#         self.fc = nn.Linear(inplane * 8, num_classes)
#         self.softmax = nn.Sigmoid()
#
#         self._initialize_weights()
#
#     def forward(self, x):
#         # Conv1D 와의 호환성을 위해 (1, 100, 512) 사이즈의 x 가 들어옴.
#         # 채널 사이즈를 1 로 받을 수 있도록 차원 추가
#         # x = x[None, :, :, :]
#         out = self.relu(self.layer1(x)  )
#         out = self.relu(self.layer2(out))
#         out = self.relu(self.layer3(out))
#         out = self.relu(self.layer4(out))
#         out = self.relu(self.layer5(out))
#         out = self.relu(self.layer6(out))
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         out = self.softmax(out)
#         return out

class Conv2D_Net(nn.Module):
    def __init__(self, inplane, num_classes=1):
        super(Conv2D_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 25088), stride=1, padding=(2, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=(5, 1), stride=1, padding=(2, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=(5, 1), stride=1, padding=(2, 0)),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(inplane * 8, num_classes)
        )
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        # Conv1D 와의 호환성을 위해 (1, 100, 512) 사이즈의 x 가 들어옴.
        # 채널 사이즈를 1 로 받을 수 있도록 차원 추가
        out = self.layer1(x, )
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.softmax(out)
        return out

    # def _initialize_weights(self):
    #         init.orthogonal_(self.layer1.weight.data, init.calculate_gain('relu'))
    #         init.orthogonal_(self.layer2.weight, init.calculate_gain('relu'))
    #         init.orthogonal_(self.layer3.weight, init.calculate_gain('relu'))
    #         init.orthogonal_(self.layer4.weight, init.calculate_gain('relu'))
    #         init.orthogonal_(self.layer5.weight, init.calculate_gain('relu'))
    #         init.orthogonal_(self.layer6.weight, init.calculate_gain('relu'))
    #         init.orthogonal_(self.fc.weight)
    #