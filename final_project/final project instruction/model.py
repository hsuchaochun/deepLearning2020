import torch
torch.cuda.current_device()
import torch.nn as nn
from torchvision import datasets ,transforms
import torchvision
from matplotlib import pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tensorflow import summary
from torchvision.utils import make_grid
import os



#Load data set
train_data = datasets.ImageFolder(
    'data path',
    transform = transforms.Compose([transforms.ToTensor()])                         
)

test_data = datasets.ImageFolder(
    'data path',
    transform = transforms.Compose([transforms.ToTensor()])                         
)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=50,shuffle= True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,shuffle=True)



#show labels
print(train_data.classes)
print(train_data.class_to_idx)

#check availability of gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



"""""

This Part You Have To Build Your Own Model

For example:
class CNN_Model(nn.Module)





And at the end, you have to save your trained weights to pth format
you can use function torch.save()
"""""






