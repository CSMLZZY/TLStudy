import os

import torch.backends.cudnn as cudnn
import torch.utils.data
from dataset.data_loader_mslf import GetLoader
from torchvision import datasets
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(dataset_name, epoch):
    assert dataset_name in ['MNIST', 'mnist_m']

    model_root = os.path.join('..', 'models')
    image_root = os.path.join('..', 'dataset', dataset_name)

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0

    # 加载数据

    # 串联多个图片变换的操作
    img_transform_source = transforms.Compose([
        # 将图片短边缩放至 image_size，长宽比保持不变
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # 为图片的三个通道分别指定均值与标准差
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    if dataset_name == 'mnist_m':
        test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')

        dataset = GetLoader(
            data_root=os.path.join(image_root, 'mnist_m_test'),
            data_list=test_list,
            transform=img_transform_target
        )
    else:
        dataset = datasets.MNIST(
            root='../dataset',
            # train (bool, optional): If True, creates dataset from ``training.pt``,
            # otherwise from ``test.pt``.
            train=False,
            transform=img_transform_source
        )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # 进行test

    my_net = torch.load(os.path.join(model_root, 'mnist_mnistm_model_epoch_' + str(epoch) + '.pth'))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.to(device)

    len_dataloader = len(dataloader)
    data_traget_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:
        # 用测试数据测试模型
        data_target = data_traget_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.to(device)
            t_label = t_label.to(device)
            input_img = input_img.to(device)
            class_label = class_label.to(device)

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)

        class_output, _ = my_net(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    # 计算准确率
    accu = n_correct.data.numpy() * 1.0 / n_total

    print('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))
