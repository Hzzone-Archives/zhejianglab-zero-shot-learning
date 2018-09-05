import numpy as np
import os
import torch
from src.model import *
from torchvision import transforms
from skimage import io, color




class LabelObject(object):

    def __init__(self, datafilepath, labelfilepath, attributes_per_class_file, class_wordembeddings_file):
        labels = np.loadtxt(labelfilepath, dtype=str)
        attributes_per_class = np.loadtxt(attributes_per_class_file, dtype=str)
        class_wordembeddings = np.loadtxt(class_wordembeddings_file, dtype=str)
        self.data = np.loadtxt(datafilepath, dtype=str)
        self.labels = {}
        self.label_names = {}
        self.attributes = {}
        self.class_wordembeddings = {}

        for i in range(labels.shape[0]):
            self.labels[labels[i, 0]] = i
            self.label_names[labels[i, 0]] = labels[i, 1]
        for i in range(attributes_per_class.shape[0]):
            class_attributes = np.reshape(np.array(list(map(float, attributes_per_class[i, 1:]))), (1, -1)).astype(np.float32)
            self.attributes[attributes_per_class[i, 0]] = class_attributes
        for i in range(class_wordembeddings.shape[0]):
            class_wordembedding = np.reshape(np.array(list(map(float, class_wordembeddings[i, 1:]))), (1, -1)).astype(np.float32)
            self.class_wordembeddings[class_wordembeddings[i, 0]] = class_wordembedding

    def get_class_wordembeddings(self):
        matrix = []
        for name in self.label_names.keys():
            matrix.append(self.class_wordembeddings[self.label_names[name]])

        return np.squeeze(np.array(matrix), axis=1)


def output_features(root, model, save_path="./tmp"):
    test_transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4814507, 0.45002526, 0.39907682), (0.27025667, 0.26409653, 0.27254263))
    ])
    for root, dirs, files in os.walk(root):
        for fname in files:
            fpath = os.path.join(root, fname)
            im = io.imread(fpath)
            ## exists some gray images
            if len(im.shape) != 3:
                im = color.gray2rgb(im)
            im = test_transformer(im)
            output = model(im.unsqueeze(0))
            torch.save(output, os.path.join(save_path, fname))
            print(os.path.join(save_path, fname))

def output_results(model1, model2, data_root, save_path):
    model1.eval()
    model2.eval()
    test_transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4814507, 0.45002526, 0.39907682), (0.27025667, 0.26409653, 0.27254263))
    ])
    root = os.path.dirname(data_root)
    label_list = np.loadtxt(os.path.join(root, "label_list.txt"), dtype=str)
    class_wordembeddings = np.loadtxt(os.path.join(root, "class_wordembeddings.txt"), dtype=str)
    labels = {}
    for i in range(label_list.shape[0]):
        for j in range(class_wordembeddings.shape[0]):
            if label_list[i, 1] == class_wordembeddings[j, 0]:
                labels[label_list[i, 0]] = np.reshape(np.array(list(map(float, class_wordembeddings[i, 1:]))), (1, -1)).astype(np.float32)
    results = ""
    target = np.array(list(labels.values())).squeeze(1)
    for root, dirs, files in os.walk(data_root):
        for fname in files:
            fpath = os.path.join(root, fname)
            im = io.imread(fpath)
            ## exists some gray images
            if len(im.shape) != 3:
                im = color.gray2rgb(im)
            im = test_transformer(im)
            output = model1(im.unsqueeze(0)).detach()
            output = model2(output)
            outputs_numpy = output.cpu().data.numpy()
            mul_value = np.dot(outputs_numpy, target.T)
            denorm1 = np.reshape(np.linalg.norm(outputs_numpy, axis=1), (-1, 1))
            denorm2 = np.reshape(np.linalg.norm(target, axis=1), (1, -1))
            denorm = np.dot(denorm1, denorm2)
            cos_value = mul_value / denorm
            tmp = np.reshape(np.argmax(cos_value, axis=1), (1, -1))
            class_number = np.asscalar(tmp)
            print(class_number)
            results += "{}\t{}\n".format(fname, list(labels.keys())[class_number])

    with open(save_path, "w") as f:
        f.write(results)

if __name__ == "__main__":

    # model_ft = torch.load("resnet18.pkl", map_location='cpu')
    # featuresModel = FeaturesNet(model_ft)
    # output_features("../data/DatasetA_train/train", featuresModel)
    model_ft = torch.load("resnet18.pkl", map_location='cpu')
    featuresModel = FeaturesNet(model_ft)
    # DEM_model = DEMNet(featuresModel, 512, 300)
    ZSL_Net = ZSLNet(512, 300)
    ZSL_Net.load_state_dict(torch.load("../models/ZSLNet.pkl"))
    output_results(featuresModel, ZSL_Net, "../data/DatasetA_test/test", "submit.txt")

