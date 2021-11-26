import torch.nn as nn
from functions_mslf import ReverseLayerF


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # 特征提取器
        self.feature = nn.Sequential()
        # Conv2d二维卷积方法  nn.Conv2d(输入的通道数目,输出的通道数目,kernel_size=卷积核的大小(当卷积是方形的时候，只需要一个整数边长即可，卷积不是方形，要输入一个元组表示 高和宽。))
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        # nn.MaxPool2d(2) 中的2是指：max pooling的窗口大小 即 2*2
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        # 参数inplace=True：inplace为True，将会改变输入的数据，否则不会改变原输入，只会产生新的输出
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        # 标签预测器
        self.class_classifier = nn.Sequential()
        # 全连接层 第一个参数为in_features：输入的二维张量的大小；第二个参数为out_features：输出的二维张量的大小
        # 从输入输出的张量的shape角度来理解，相当于一个输入为[batch_size, in_features]的张量变换成了[batch_size, out_features]的输出张量。
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        # 域分类器
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        # 将每条数据扩展为(3,28,28)的形状，共有.shape[0]条数据
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        # 调用的是ReverseLayerF中的forward函数
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
