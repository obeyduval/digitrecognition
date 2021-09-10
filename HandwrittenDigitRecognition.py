# Handwritten Digit Recognition Using PyTorch

# Pytorch Libraries and MNIST Database

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
# val set is what the network will be tested again on

trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# images and labels (pull in one batch at a time)
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.shape)
print(labels.shape)

# display one image
# plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
# plt.show()

# display multiple images
# figure = plt.figure()
# num_of_images = 60
# for index in range(1, num_of_images + 1):
#     plt.subplot(6, 10, index)
#     plt.axis('off')
#     plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
# plt.show()

# NETWORK IS INPUT LAYER, 2 HIDDEN LAYERS, and OUTPUT LAYER

input_size = 784  # input layer
hidden_sizes = [128, 64]  # hidden layers
output_size = 10  # output layer

# set up model
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print(model)

# criterion is the negative log likelihood loss
criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images)  # log probabilities
loss = criterion(logps, labels)  # calculate the NLL loss

# weights before backward prop
print('Before backward pass: \n', model[0].weight.grad)
loss.backward()

# weights after back prop
print('After backward pass: \n', model[0].weight.grad)

# training and updating weights
