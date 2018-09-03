import numpy as np
import os
import torch



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

