# Handwritten Digit Recognition Using PyTorch

# Pytorch Libraries and MNIST Database

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

# toTensor - converts the image into numbers
# normalize - normalize tensor with a mean and stdv

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])

# batch size is the number of images we read in on one go
# train set is what the network is trained on
# val set is what the network will be tested on

trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# images and labels
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.shape)
print(labels.shape)


