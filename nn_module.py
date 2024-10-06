# This is a sample Python script.

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
from typing import List

import torchvision
from main import Visualize

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    root_path = os.path.dirname(__file__)
    train_set = torchvision.datasets.CIFAR10(root="./dataset/torchvision", train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root="./dataset/torchvision", train=False, download=True)

    visual_tool = Visualize(log_dir="p10")
    for i in range(10):
        img, target = test_set[i]
        visual_tool.add_pil_img(img)
    visual_tool.visualize_using_tensor()
    visual_tool.visualize_img()
    pass


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
