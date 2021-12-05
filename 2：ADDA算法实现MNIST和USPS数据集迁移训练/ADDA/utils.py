# ok
# Utilities for ADDA.

import os
import random

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import params
from datasets import get_mnist, get_usps


# 将张量转换为变量
def make_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


# 反转归一化，然后将数组转换为图像
def denormalize(x, std, mean):
    out = x * std + mean
    return out.clamp(0, 1)


# 初始化权重
def init_weights(layer):
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


# 初始化随机种子
def init_random_seed(manual_seed):
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 按名称获取数据加载器
def get_data_loader(name, train=True):
    if name == "MNIST":
        return get_mnist(train)
    elif name == "USPS":
        return get_usps(train)


# 使用 cuda 和权重初始化模型
def init_model(net, restore):
    # 初始化权重
    net.apply(init_weights)

    # 恢复模型权重
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # 检查 cuda 是否可用
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net


# 保存模型
def save_model(net, filename):
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(),
               os.path.join(params.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(params.model_root,
                                                             filename)))
