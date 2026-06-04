'''
# @Time    : 2025/5/14 9:34
# @Author  : Fengcheng Ji
# @Function: 遥感图像块数据集，加载 .npy 格式数据，提取中心像素标签
'''
import numpy as np
import torch
from torch.utils.data import Dataset


class RemoteSensingDataset(Dataset):
    def __init__(self, image_path, label_path):
        """
        image_path: 图像块 .npy 文件，形状 [num, w, h, channel]
        label_path: 标签 .npy 文件，形状 [num, w, h, class]
        """
        images = np.load(image_path)   # [num, w, h, channel]
        labels = np.load(label_path)   # [num, w, h, class]

        images = images.transpose(0, 3, 1, 2)  # [num, channel, w, h]
        images = images / 10000

        half_w, half_h = labels.shape[1] // 2, labels.shape[2] // 2
        center_labels = labels[:, half_w, half_h, :]  # [num, class]
        center_labels = np.argmax(center_labels, axis=1)  # [num]

        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(center_labels).long()

        self.in_channels = self.images.shape[1]
        self.num_classes = len(torch.unique(self.labels))
        print('xxx')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
