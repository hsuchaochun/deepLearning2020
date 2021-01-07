#! /usr/bin/env python

import torch
import torch.nn as nn
from torchvision import datasets ,models,transforms
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import torchvision

#load image from folder and set foldername as label
train_data = datasets.ImageFolder(
    '/home/austin/trailnet-testing-Pytorch/duckiefloat_line_follow/src/data/Angle_train',
    transform = transforms.Compose([transforms.ToTensor()])                         
)

test_data = datasets.ImageFolder(
    '/home/austin/trailnet-testing-Pytorch/duckiefloat_line_follow/src/data/Angle_test',
    transform = transforms.Compose([transforms.ToTensor()])                         
)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=40,shuffle= True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=40,shuffle=True)

#CNN model
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Sequential(              
            nn.Conv2d(
                in_channels=3,              
                out_channels=32,            
                kernel_size=4,              
                stride=1,                   
                padding=0,                  
            ),                                                 
            nn.MaxPool2d(kernel_size=2, stride=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=0,
            ),                           
            nn.MaxPool2d(kernel_size=2, stride=2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=0,
            ),                           
            nn.MaxPool2d(kernel_size=2, stride=2),                
        )
        self.conv4 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=1,
            ),                           
            nn.MaxPool2d(kernel_size=2, stride=2),                
        )
        self.fc1 = nn.Linear(38528, 200)
        self.fc2 = nn.Linear(200, 13)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

net = CNN_Model().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(100): # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the input
        inputs, labels = data

        # wrap time in Variable
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 0:   # print every 2000 mini-batches
            print('[%d, %d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')
torch.save(net.state_dict(),'/home/austin/trailnet-testing-Pytorch/duckiefloat_line_follow/src/line_detect/src/line_angle.pth')

#Accuracy present
print('Accuracy testing...')
correct = 0
total = 0
for data in test_loader:
    images, labels = data
    images = images.cuda()
    labels = labels.cuda()
    with torch.no_grad():
         outputs = net(Variable(images))
         _,predicted = torch.max(outputs.data,1)
    #print('predict:',predicted)
    #print('labels.:',labels)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
