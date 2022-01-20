# ok
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pylab
import params


# 可视化loss
def visulize_loss(train_hist):
    x = range(len(train_hist['Total_loss']))
    x = [i * params.plot_iter for i in x]

    total_loss = train_hist['Total_loss']
    class_loss = train_hist['Class_loss']
    mmd_loss = train_hist['MMD_loss']

    plt.plot(x, total_loss, label='total loss')
    plt.plot(x, class_loss, label='class loss')
    plt.plot(x, mmd_loss, label='mmd loss')

    plt.xlabel('Step')
    plt.ylabel('Loss')

    plt.grid(True)
    pylab.show()


# 可视化accuracy
def visualize_accuracy(test_hist):
    x = range(len(test_hist['Source Accuracy']))

    source_accuracy = test_hist['Source Accuracy']
    target_accuracy = test_hist['Target Accuracy']

    plt.plot(x, source_accuracy, label='source accuracy')
    plt.plot(x, target_accuracy, label='target accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.grid(True)
    pylab.show()


# 显示图片
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    pylab.show()


def pairwise_distance(x, y):
    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output


def gaussian_kernel_matrix(x, y, sigmas):
    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost


def mmd_loss(source_features, target_features):
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    # partial函数的解析见：https://blog.csdn.net/qq_38048756/article/details/118339890
    if params.use_gpu:
        gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=Variable(torch.cuda.FloatTensor(sigmas)))
    else:
        gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=Variable(torch.FloatTensor(sigmas)))

    loss_value = maximum_mean_discrepancy(source_features, target_features, kernel=gaussian_kernel)
    loss_value = loss_value

    return loss_value


# 获取训练用的DataLoader
def get_train_loader(dataset):
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            # 修改前：transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)])
            # 由于MNIST数据集为灰度图，只有一个通道，故需要修改为：
            transforms.Normalize(mean=[0.5], std=[0.5])])
        data = datasets.MNIST(root=params.mnist_path, train=True, transform=transform, download=True)
        dataloader = DataLoader(dataset=data, batch_size=params.batch_size, shuffle=True, drop_last=True)

    elif dataset == 'MNIST_M':
        transform = transforms.Compose([
            transforms.RandomCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)])
        data = datasets.ImageFolder(root=params.mnistm_path + '/train', transform=transform)
        dataloader = DataLoader(dataset=data, batch_size=params.batch_size, shuffle=True, drop_last=True)

    else:
        raise Exception('不存在数据集：{}'.format(str(dataset)))

    return dataloader


# 获取测试用的DataLoader
def get_test_loader(dataset):
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            # 修改前：transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)])
            # 由于MNIST数据集为灰度图，只有一个通道，故需要修改为：
            transforms.Normalize(mean=[0.5], std=[0.5])])
        data = datasets.MNIST(root=params.mnist_path, train=False, transform=transform, download=True)
        dataloader = DataLoader(dataset=data, batch_size=params.batch_size, shuffle=True)

    elif dataset == 'MNIST_M':
        transform = transforms.Compose([
            transforms.CenterCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)])
        data = datasets.ImageFolder(root=params.mnistm_path + '/test', transform=transform)
        dataloader = DataLoader(dataset=data, batch_size=params.batch_size, shuffle=True)

    else:
        raise Exception('不存在数据集：{}'.format(str(dataset)))

    return dataloader
