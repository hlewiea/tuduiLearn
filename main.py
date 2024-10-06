# This is a sample Python script.

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import shutil
from typing import List, Dict
from unittest.mock import patch

from debugpy.common.log import log_dir, write
from jupyter_core.version import parts
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import cv2

from torchvision import transforms

import subprocess
import signal

root_path = os.path.dirname(__file__)
class PortProcessManager:
    def __init__(self, port):
        self.port = port

    def find_process_by_port(self):
        # 使用 netstat 查找占用指定端口的进程
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if f":{self.port}" in line:
                # 提取 PID
                parts = line.split()
                pid = parts[-1]
                return pid
        return None

    def kill_process(self):
        pid = self.find_process_by_port()
        if pid:
            try:
                # 终止进程
                os.kill(int(pid), signal.SIGTERM)
                print(f"Process {pid} terminated.")

                # 这里可以添加重启进程的逻辑，例如重新启动某个服务
                # subprocess.Popen(['your_command_here'])

            except Exception as e:
                print(f"Error terminating process {pid}: {e}")
        else:
            print(f"No process found on port {self.port}.")

    @staticmethod
    def start_background_process(new_command, working_dir):
        # 重新启动进程并传入新的参数
        subprocess.Popen(new_command, cwd=working_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Process restarted with new command: {new_command}")

class ImgVisualItem:
    def __init__(self, title, imgs=None, dataformats='CHW'):
        if imgs is None:
            imgs = []
        self.title = title
        self.dataformats = dataformats
        self.imgs = imgs

    def append(self, img):
        self.imgs.append(img)

    def __len__(self):
        return len(self.imgs)

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.path = os.path.join(root_dir, label_dir)
        self.img_path_list = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path_list[idx]
        img_item_path = os.path.join(self.path, img_name)
        return img_item_path

    def __len__(self):
        return len(self.img_path_list)

class Visualize:
    def __init__(self, log_dir="logs"):
        self.img_pil = []
        self.img_np = []
        self.imgs:Dict[str, ImgVisualItem] = {}
        self.log_dir = log_dir

        self.tensor_trans = transforms.ToTensor()
        self.trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.8, 0.5, 0.5])
        self.trans_resize = transforms.Resize((512, 512))
        self.trans_resize_2 = transforms.Resize(256)
        self.trans_compose = transforms.Compose([self.trans_resize_2, self.tensor_trans])

        self.trans_random = transforms.RandomCrop((256, 256))
        self.trans_compose_2 = transforms.Compose([self.trans_random, self.tensor_trans])

    def add_one_img(self, img_path=""):
        self.img_pil.append(self.get_pil_img(img_path))
        self.img_np.append(self.get_np_img(img_path))

    def add_pil_img(self, img):
        self.img_pil.append(img)
        self.img_np.append(self.get_np_img(img))

    @staticmethod
    def get_pil_img(image_path):
        return Image.open(image_path)

    @staticmethod
    def get_np_img(img_pil):
        return np.array(img_pil)

    def visualize_img(self, multi_img=False, is_clean=False):
        # tensorboard 可视化
        # tensorboard --logdir logs --port xxx
        if is_clean:
            self.clean_dir()
        writer = SummaryWriter(self.log_dir)
        for _, img_items in self.imgs.items():
            title = img_items.title
            dataformats = img_items.dataformats
            for idx, img in enumerate(img_items.imgs):
                if multi_img:
                    writer.add_images(title, img, idx)
                else:
                    writer.add_image(title, img, idx, dataformats=dataformats)
        writer.close()
        self.manage_process()

    def visualize_graph(self, model, sequence, is_clean=False):
        # tensorboard 可视化
        # tensorboard --logdir logs --port xxx
        if is_clean:
            self.clean_dir()
        writer = SummaryWriter(self.log_dir)
        writer.add_graph(model, sequence)
        writer.close()
        self.manage_process()

    def clean_dir(self):
        log_dir = os.path.join(root_path, self.log_dir)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            os.makedirs(log_dir)

    def manage_process(self, port=6555):
        command = f"tensorboard --logdir {self.log_dir} --port {port}"
        manager = PortProcessManager(port)
        manager.kill_process()
        manager.start_background_process(command, working_dir=root_path)

    def visualize_function(self):
        # tensorboard 可视化
        writer = SummaryWriter(self.log_dir)
        for i in range(1, 100):
            writer.add_scalar('Y=X^2', i * i, i)
        writer.close()

    def tensor_origin(self, key, tensor_img):
        self.__insert_img__(key, tensor_img)
        return tensor_img

    def to_tensor_origin(self, img_np):
        # __call__()
        tensor_img = self.tensor_trans(img_np)
        self.__insert_img__("Tensor", tensor_img)
        return tensor_img

    def __insert_img__(self, key, tensor_img):
        if key not in self.imgs:
            self.imgs[key] = ImgVisualItem(key, [tensor_img])
            return
        self.imgs[key].append(tensor_img)

    def tensor_normalize(self, tensor_img):
        # normalize
        # output[channel] = (input[channel] - mean[channel]) / std[channel]
        img_norm = self.trans_norm(tensor_img)
        self.__insert_img__("Normalize", img_norm)

    def tensor_resize(self, img_pil):
        # resize
        img_resize = self.trans_resize(img_pil)
        img_resize = self.tensor_trans(img_resize)
        self.__insert_img__("Resize", img_resize)

    def tensor_compose_resize(self, img_pil):
        # compose - resize
        img_resize_2 = self.trans_compose(img_pil)
        self.__insert_img__("Compose_resize", img_resize_2)

    def tensor_random_crop(self, img_pil, steps=10):
        # RandomCrop
        for i in range(steps):
            self.__insert_img__("RandomCrop", self.trans_compose_2(img_pil))

    def visualize_using_tensor(self):
        for idx in range(len(self.img_pil)):
            img_pil = self.img_pil[idx]
            img_np = self.img_np[idx]

            tensor_img = self.to_tensor_origin(img_np)
            # self.tensor_normalize(tensor_img)
            # self.tensor_resize(img_pil)
            # self.tensor_compose_resize(img_pil)



    @staticmethod
    def opencv_img(path):
        return cv2.imread(path)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset_path = os.path.join(root_path, "dataset", "hymenoptera_data", "train")
    ant_dataset = MyData(dataset_path, 'ants')
    bees_dataset = MyData(dataset_path, 'bees')

    # ant_dataset.visualize_using_np(1)
    idx = 1
    img = ant_dataset[idx]
    visual_tool = Visualize()
    visual_tool.add_one_img(img)
    visual_tool.visualize_using_tensor()
    # cv_img = visual_tool.opencv_img()
    visual_tool.visualize_img()
    pass


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
