from __future__ import print_function, division
import os
from skimage import io, color
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import torch
from .utils import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class ImageDataset(Dataset):
    def __init__(self, data_root, datafilepath, labelfilepath, attributes_per_class_file, class_wordembeddings_file, transform=None):
        self.labelObject = LabelObject(datafilepath, labelfilepath, attributes_per_class_file, class_wordembeddings_file)
        self.dataroot = data_root
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataroot, self.labelObject.data[idx, 0])
        label_name = self.labelObject.data[idx, 1]
        label = self.labelObject.labels[label_name]
        image = io.imread(img_path)
        wordembeddings = self.labelObject.class_wordembeddings[self.labelObject.label_names[label_name]]
        attributes = self.labelObject.attributes[label_name]

        ## exists some gray images
        if len(image.shape) != 3:
            image = color.gray2rgb(image)

        # print(image.shape)
        # sample = {"image": image, "label": label}
        if self.transform:
            image = self.transform(image)

        return image, label, attributes, wordembeddings

    def __len__(self):
        return self.labelObject.data.shape[0]

class VectorDataset(Dataset):
    def __init__(self, data_root, datafilepath, labelfilepath, attributes_per_class_file, class_wordembeddings_file):
        self.labelObject = LabelObject(datafilepath, labelfilepath, attributes_per_class_file, class_wordembeddings_file)
        self.dataroot = data_root

    def __getitem__(self, idx):
        featurs_path = os.path.join(self.dataroot, self.labelObject.data[idx, 0])
        label_name = self.labelObject.data[idx, 1]
        label = self.labelObject.labels[label_name]
        # image = io.imread(img_path)
        features = torch.load(featurs_path)
        wordembeddings = self.labelObject.class_wordembeddings[self.labelObject.label_names[label_name]]
        attributes = self.labelObject.attributes[label_name]

        return features.detach().numpy(), label, attributes, wordembeddings

    def __len__(self):
        return self.labelObject.data.shape[0]

def getTransforms():

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Scale(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4814507, 0.45002526, 0.39907682), (0.27025667, 0.26409653, 0.27254263))
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4814507, 0.45002526, 0.39907682), (0.27025667, 0.26409653, 0.27254263))
        ]),
    }

    return data_transforms

def getImageLoaders(root, BATCH_SIZE=120):

    data_transforms = getTransforms()

    image_datasets = {x: ImageDataset(root+"train", root+x+".txt", root+"label_list.txt", root+"attributes_per_class.txt", root+"class_wordembeddings.txt", transform=data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                  shuffle=True if x == "train" else False, num_workers=4)
                   for x in ['train', 'val']}
    return dataloaders


def getVectorLoaders(root, BATCH_SIZE=120):

    vector_datasets = {x: VectorDataset("./tmp", root+x+".txt", root+"label_list.txt", root+"attributes_per_class.txt", root+"class_wordembeddings.txt")
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(vector_datasets[x], batch_size=BATCH_SIZE,
                                                  shuffle=True if x == "train" else False, num_workers=4)
                   for x in ['train', 'val']}
    return dataloaders
