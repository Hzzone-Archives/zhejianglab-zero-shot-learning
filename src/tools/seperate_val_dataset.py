import os.path as osp
import numpy as np
import random
import shutil

root = "../../data/DatasetA_train"

whole_data = {}

whole_data_arr = np.loadtxt(osp.join(root, "train.txt"), dtype=str)

for i in range(whole_data_arr.shape[0]):
    filename = whole_data_arr[i, 0]
    label = whole_data_arr[i, 1]
    if not label in whole_data.keys():
        whole_data[label] = []
    whole_data[label].append(filename)

val_data = []
train_data = []
for key in whole_data.keys():
    value = whole_data[key]
    random.shuffle(value)
    random.shuffle(value)
    val_data.extend([[x, key] for x in value[:len(value)//5]])
    train_data.extend([[x, key] for x in value[len(value)//5:]])


val_data = np.array(val_data).reshape((-1, 2))
np.savetxt(osp.join(root, "val.txt"), val_data, fmt="%s")
train_data = np.array(train_data).reshape((-1, 2))
shutil.copy(osp.join(root, "train.txt"), osp.join(root, "backup_train.txt"))
np.savetxt(osp.join(root, "train.txt"), train_data, fmt="%s")
