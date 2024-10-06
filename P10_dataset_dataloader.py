# This is a sample Python script.

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
from typing import List

import torchvision
from torch.utils.data import DataLoader

from main import Visualize

def load_pil_imgs(dataset):
    visual_tool = Visualize(log_dir="p10")
    for i in range(10):
        img, target = dataset[i]
        visual_tool.add_pil_img(img)
    visual_tool.visualize_using_tensor()
    visual_tool.visualize_img()

def load_tensor_img(data_loader):
    visual_tool = Visualize(log_dir="dataloader")
    for epoch in range(2):
        for data in data_loader:
            img, target = data
            visual_tool.tensor_origin(f"Test_data_Epoch_shuffle {epoch}", img)

    visual_tool.visualize_img(multi_img=True, is_clean=True)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    root_path = os.path.dirname(__file__)
    # 训练集+测试集
    train_set = torchvision.datasets.CIFAR10(root="./dataset/torchvision", train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root="./dataset/torchvision", train=False, download=True)
    load_pil_imgs(test_set)

    # 训练集处理
    test_data = torchvision.datasets.CIFAR10(root="./dataset/torchvision", train=True,
                                             transform=torchvision.transforms.ToTensor())

    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=4, drop_last=True)

    load_tensor_img(test_loader)

    pass


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
