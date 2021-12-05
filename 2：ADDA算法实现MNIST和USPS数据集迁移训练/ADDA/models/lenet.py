# ok
# ADDA的LeNet模型

import torch.nn.functional as F
from torch import nn


# ADDA 的 LeNet 编码器模型
class LeNetEncoder(nn.Module):
    # 初始化LeNet编码器
    def __init__(self):
        super(LeNetEncoder, self).__init__()
        self.restored = False
        self.encoder = nn.Sequential(
            # 1st conv layer
            # input [1 x 28 x 28]
            # output [20 x 12 x 12]
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 2nd conv layer
            # input [20 x 12 x 12]
            # output [50 x 4 x 4]
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(50 * 4 * 4, 500)

    def forward(self, input):
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
        return feat


# ADDA 的 LeNet 分类器模型
class LeNetClassifier(nn.Module):
    # 初始化LeNet分类器
    def __init__(self):
        super(LeNetClassifier, self).__init__()
        self.fc2 = nn.Linear(500, 10)

    def forward(self, feat):
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out
