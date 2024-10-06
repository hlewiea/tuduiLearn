# This is a sample Python script.

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import time
from typing import List

import torch
import torchvision
from mpmath.identification import transforms
from numpy.core.defchararray import title
from pycparser.ply.yacc import resultlimit
from pygments.util import tag_re
from sympy.physics.vector import outer
from torch.onnx.symbolic_opset9 import tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.video.resnet import model_urls
from urllib3.filepost import writer

from main import Visualize
from torch import nn
import torch.nn.functional as F
import random
from P10_dataset_dataloader import load_tensor_img, load_tensor_graph
from PIL import Image

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()
        self.linear1 = nn.Linear(32*32*3*64, 10)

        self.model1 = nn.Sequential(
            # 3 @ 32 @ 32 -> 32 @ 32 @ 32 ;  padding 和 (kernel_size - 1) / 2 抵消
            nn.Conv2d(3, 32, 5 , 1, 2),
            # 32 @ 32 @ 32 -> 32 @ 16 @ 16
            nn.MaxPool2d(2),
            #  32 @ 16 @ 16 ->  32 @ 16 @ 16
            nn.Conv2d(32, 32, 5, 1, 2),
            #  32 @ 16 @ 16 ->  32 @ 8 @ 8
            nn.MaxPool2d(2),
            #  32 @ 8 @ 8 ->  64 @ 8 @ 8
            nn.Conv2d(32, 64, 5, 1, 2),
            #  64 @ 8 @ 8 ->  64 @ 4 @ 4
            nn.MaxPool2d(2),
            # 64 @ 4 @ 4 -> 64 * 4 * 4
            nn.Flatten(),
            # 64
            nn.Linear(64 * 4 * 4, 64),
            # 10
            nn.Linear(64, 10)
        )

    def forward(self, input):
        # output = self.conv1(input)
        # output = self.maxpool1(input)
        # output = self.relu1(input)
        # output = self.sigmoid1(input)

        # output = self.linear1(input)
        # output = torch.reshape(output, [1, 1, 1, -1])
        # output = torch.flatten(output)

        # if output.size()[1] != 3:
        #     output = torch.reshape(output, [-1, 3, -1, -1])

        output = self.model1(input)
        return output

    def test(self):
        # 64 个数据
        input = torch.ones([64, 3, 32, 32])
        output = self.forward(input)
        print(output.size())

def generate_random_matrix(n, m, lower_bound, upper_bound):
    """生成一个 n x m 的随机整数矩阵"""
    return [[random.randint(lower_bound, upper_bound) for _ in range(m)] for _ in range(n)]

def test_conv():
    # 全1初始化
    torch.ones([64,3,32,32])
    input_matrix = torch.tensor(generate_random_matrix(5, 5, 0, 9), dtype=torch.float)
    kernel_matrix = torch.tensor(generate_random_matrix(3, 3, 0, 9), dtype=torch.float)
    print(input_matrix)
    print(kernel_matrix)
    input = torch.reshape(input_matrix, (1, 1, 5, 5))
    kernel = torch.reshape(kernel_matrix, [1,1,3,3])
    print(input)
    print(kernel)
    # 图卷积
    output = F.conv2d(input, kernel, stride=1)
    print(output)
    # 图卷积
    output = F.conv2d(input, kernel, stride=2)
    print(output)
    # padding + 卷积
    # 图卷积
    output = F.conv2d(input, kernel, stride=1, padding=1)
    print(output)

def test_img_conv(test_loader):
    my_nn = MyModule()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(my_nn.parameters(), lr=0.01)
    processed_data = []
    for epoch in range(20):
        running_loss = 0.0
        for data in test_loader:
            imgs, target = data
            output = my_nn(imgs)
            result_loss = loss(output, target)
            # print(target)
            # print(output)
            # print(result_loss)
            optimizer.zero_grad()
            result_loss.backward()
            optimizer.step()
            running_loss += result_loss
            # processed_data.append((output, target))
        print(running_loss)

    # load_tensor_img(test_loader, title="test_loader", is_clean=True)
    # load_tensor_img(processed_data, title="processed_data")

def test_graph():
    my_nn = MyModule()
    input = torch.ones([64, 3, 32, 32])
    output = my_nn(input)
    print(output.size())
    load_tensor_graph(my_nn, input)

def test_loss_function():
    inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
    targets = torch.tensor([1, 2, 5], dtype=torch.float32)
    inputs = torch.reshape(inputs, (1,1,1,3))
    targets = torch.reshape(targets, (1,1,1,3))

    loss = nn.L1Loss()
    result = loss(inputs, targets)
    print(result)

    loss_mse = nn.MSELoss()
    result_mse = loss_mse(inputs, targets)
    print(result_mse)

    # crossEntropy
    x = torch.tensor([0.1, 0.2, 0.3])
    y = torch.tensor([1])
    x = torch.reshape(x, (1, 3))
    loss_cross = nn.CrossEntropyLoss()
    result_cross = loss_cross(x, y)
    print(result_cross)

def handle_model_from_torch():
    # test_data = torchvision.datasets.ImageNet(root="./dataset/data_image_net", train=True, download=True, transform=torchvision.transforms.ToTensor())
    # 模型下载
    vgg16_false = torchvision.models.vgg16(pretrained=False)
    vgg16_true = torchvision.models.vgg16(pretrained=True)
    print(vgg16_true)

    # 模型修改
    vgg16_true.classifier.add_module("add_linear_in_classifier", nn.Linear(1000, 10))
    print(vgg16_true)
    vgg16_true.add_module("add_linear", nn.Linear(10, 3))
    print(vgg16_true)
    print("=" * 100)
    print(vgg16_false)
    vgg16_false.classifier[6] = nn.Linear(4096, 10)
    print(vgg16_false)

    # 模型保存
    # 方法1: 保存网络模型的结构+参数
    torch.save(vgg16_true, "vgg16_method1.pth")
    # 方法2: 保存网络模型的参数 to dict (recommend)
    torch.save(vgg16_true.state_dict(), "vgg16_method2.pth")

    # 模型加载
    # 方法1
    model = torch.load("vgg16_method1.pth")
    # 方法2
    vgg16 = torchvision.models.vgg16(pretrained=False)
    vgg16.load_state_dict(torch.load("vgg16_method2.pth"))

    # 方法一的陷阱. 使用前需要先导入自定义的模型类（我理解也不是陷阱，加载在线模型时也需要先导入模型。 ）
    my_nn = MyModule()
    torch.save(my_nn, "my_nn_100.pth")
    my_model = torch.load("my_nn_100.pth")

def handle_model_training(train_loader:DataLoader, test_loader:DataLoader, tarin_size, test_size, device=None, cuda_method=False):
    my_nn = MyModule()
    # method1: 模型/loss函数/训练/测试数据集
    if device: # 模型无需重复复制 .to即可
        my_nn = my_nn.to(device)
    if cuda_method :
        my_nn = my_nn.cuda()
    # my_nn.test()
    # my_nn = torch.load("my_nn_100.pth")
    # 损失函数
    loss_cross = nn.CrossEntropyLoss()
    if device:
        loss_cross = loss_cross.to(device)
    if cuda_method == 1:
        loss_cross = loss_cross.cuda()
    # 优化器
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(my_nn.parameters(), lr=learning_rate)

    # 训练次数
    epoch_start = 0
    epoch = 20
    visual = Visualize("train")
    writer = SummaryWriter("train")
    # 记录训练次数
    total_train_step = epoch_start
    total_test_step = epoch_start

    my_nn.train()
    for i in range(epoch_start, epoch):
        print(f"----------the {i+1}th train start------------")
        for idx, data in enumerate(train_loader):
            imgs, targets = data
            if device:
                imgs = imgs.to(device)
                targets = targets.to(device)
            if cuda_method:
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = my_nn(imgs)
            loss = loss_cross(outputs, targets)
            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_step += 1
            if total_train_step % 100 == 0:
                print(f"batch{idx}, 训练次数{total_train_step}, loss: {loss.item()}")
                writer.add_scalar("train_loss", loss.item(), total_train_step)

        my_nn.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_loader:
                imgs, targets = data
                if device:
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                if cuda_method:
                    imgs = imgs.cuda()
                    targets = targets.cuda()
                outputs = my_nn(imgs)
                loss = loss_cross(outputs, targets)
                total_test_loss += loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy += accuracy
        total_test_step += 1
        print(f"整体测试集上的loss： {total_test_loss}\n \t 正确率: {total_accuracy/test_size}")
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy/test_size, total_test_step)
        torch.save(my_nn, f"my_nn_100.pth")

    writer.close()
    visual.manage_process()

def test_classification():
    outputs = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    preds = outputs.argmax(1)
    print(preds)
    targets = torch.tensor([0, 1])
    print(preds == targets.sum())

def handle_model_testing(device):
    image_path = "./test_dog.png"
    image = Image.open(image_path)
    print(image)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor()
    ])
    image = transform(image)
    print(image)

    # model_= torch.load("./my_nn_100.pth", weights_only=False, map_location=torch.device("cpu"))
    model = torch.load("./my_nn_100.pth", weights_only=False).to(device)
    image = torch.reshape(image, (1, 3, 32, 32))
    model.eval()
    with torch.no_grad():
        output = model(image.to(device))
    print(output)
    print(output.argmax(1))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    root_path = os.path.dirname(__file__)
    # 测试数据集
    train_data = torchvision.datasets.CIFAR10(root="./dataset/torchvision", train=True,
                                             transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.CIFAR10(root="./dataset/torchvision", train=False,
                                             transform=torchvision.transforms.ToTensor())
    test_loader = DataLoader(dataset=test_data, batch_size=64)
    train_loader = DataLoader(dataset=test_data, batch_size=64)
    # 定义训练设备
    # device = torch.device("cpu")
    # device = torch.device("cuda:0") # 50s 单显卡
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 30s

    # test_img_conv(test_loader)
    # test_graph()
    # test_loss_function()
    # handle_model_from_torch()
    # test_classification()
    # start = time.time() # 29.8796s
    # handle_model_training(train_loader, test_loader, len(train_data), len(test_data), device, True)
    # print(time.time() - start)

    # start = time.time() # 115.0064s
    # handle_model_training(train_loader, test_loader, len(train_data), len(test_data), device)
    # print(time.time() - start)

    # 第二种使用device方法
    # start = time.time()
    # handle_model_training(train_loader, test_loader, len(train_data), len(test_data), device)
    # print(time.time() - start)

    # 测试流程
    handle_model_testing(device)
    pass





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
