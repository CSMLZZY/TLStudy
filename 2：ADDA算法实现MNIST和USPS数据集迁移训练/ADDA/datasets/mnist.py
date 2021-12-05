# ok
# MNIST 的数据集设置和数据加载器
import torch
from torchvision import datasets, transforms
import params


# 获取MNIST数据集加载器
def get_mnist(train):
    # 图像预处理
    # 关于transforms.ToTensor()函数的解析见以下链接：
    # https: // www.pianshen.com / article / 6972192583 /
    """
    transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)
    """
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      # 逐通道的对图像进行标准化（此处的图片有三个通道）
                                      transforms.Normalize(
                                          mean=[0.5],
                                          std=[0.5])])

    # 数据集和数据加载器
    mnist_dataset = datasets.MNIST(root=params.data_root,
                                   train=train,
                                   transform=pre_process,
                                   download=True)

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return mnist_data_loader
