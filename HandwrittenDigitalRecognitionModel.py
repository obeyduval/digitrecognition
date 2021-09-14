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

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])

valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

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

model.load_state_dict(torch.load('C:/Users/C22Timothy.Jackson/Downloads/digitrecognition/my_mnist_model.pt'))
model.eval()

badimages = []
# testing process
correct_count, all_count = 0, 0
for images, labels in valloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if (true_label == pred_label):
            correct_count += 1
        else:
            badimages.append(img)
        all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count / all_count))

# for index in range(1, 10):
#     plt.subplot(2, 5, index)
#     plt.axis('off')
#     plt.imshow(badimages[index].numpy().squeeze(), cmap='gray_r')
# plt.show()
