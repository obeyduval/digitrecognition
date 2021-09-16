# Handwritten Digit Recognition Using PyTorch
# ** need to save model

# Pytorch Libraries and MNIST Database

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from time import time
from torchvision import datasets, transforms
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,5,1,2)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16,32,5,1,2)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
       # self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = x.view(x.size(0),-1)
#        x = F.relu(self.fc1(x))
 #       x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


if __name__ == '__main__':

    # cross entropy loss (Classification Cross-Entropy loss and SGD with momentum)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train network (loop over our data iterator, and feed the inputs to the network and optimize)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            print(i)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

torch.save(net.state_dict(), './my_mnist_covolutional_model.pt')
