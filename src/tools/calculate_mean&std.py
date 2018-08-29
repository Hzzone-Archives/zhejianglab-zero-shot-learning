from skimage import io, transform
import torch
from torchvision import transforms, utils
import sys
sys.path.append('../')
from src.ImageDataset import *


mytransform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor()
])

dataset = ClassificationDataset("../../data/DatasetA_train/train", "../../data/DatasetA_train/label_list.txt", "../../data/DatasetA_train/train.txt", transform=mytransform)


dataloader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=4)

pop_mean = []
pop_std0 = []
# print(dataset)
for i, (img, label) in enumerate(dataloader):
    # print(img, label)
    print(i, label)
    # shape (batch_size, 3, height, width)
    numpy_image = img.numpy()

    # shape (3,)
    batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
    batch_std0 = np.std(numpy_image, axis=(0, 2, 3))

    pop_mean.append(batch_mean)
    pop_std0.append(batch_std0)

# shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
pop_mean = np.array(pop_mean).mean(axis=0)
pop_std0 = np.array(pop_std0).mean(axis=0)

print(pop_mean, pop_std0)


