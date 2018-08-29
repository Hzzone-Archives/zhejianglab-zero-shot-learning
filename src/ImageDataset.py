from __future__ import print_function, division
import os
from skimage import io, transform, color
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class ClassificationDataset(Dataset):
    def __init__(self, data_root, labelfilepath, datafilepath, transform=None):
        labels = np.loadtxt(labelfilepath, dtype=str)
        self.labels = {}
        for i in range(labels.shape[0]):
            self.labels[labels[i, 0]] = i
        self.data = np.loadtxt(datafilepath, dtype=str)
        self.dataroot = data_root
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataroot, self.data[idx, 0])
        label = self.labels[self.data[idx, 1]]
        image = io.imread(img_path)
        ## exists some gray images
        if len(image.shape) != 3:
            image = color.gray2rgb(image)

        # print(image.shape)
        # sample = {"image": image, "label": label}
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.data.shape[0]

# dataset =  ClassificationDataset("../data/DatasetA_train/train", "../data/DatasetA_train/label_list.txt", "../data/DatasetA_train/train.txt")
# print(len(dataset))
# plt.axis('off')
# plt.ioff()
# plt.imshow(dataset[0]["image"], )
# plt.tight_layout()
# plt.show()
