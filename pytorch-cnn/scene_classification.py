#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 20:56:55 2017

Convolutional neural networks to classify scene images categories.
The dataset used is 3000 scene images from SUN dataset split equally as
training set and testing set.

@author: putama
"""

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models

import time

# %% define the network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool  = nn.MaxPool2d(4,4)
        self.fc1   = nn.Linear(24200, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 15)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 24200)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# %% visualize images
def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

def main():
    # %% options
    pretrained = True
    pretrain_model = 'resnet'
    
    # %% set up the input dataset
    batch_size = 20
    
    transform = transforms.Compose([
                            transforms.RandomSizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                                  std = [ 0.229, 0.224, 0.225 ]),
                         ])
    
    trainset = dset.ImageFolder(root='./scene_dataset/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
    
    testset = dset.ImageFolder(root='./scene_dataset/test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)


    # show some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    # print images
    imshow(torchvision.utils.make_grid(images))
    
    # use pretrain model
    if pretrained:
        if pretrain_model == 'resnet':
            convNet = models.resnet18(pretrained=True)
        elif pretrain_model == 'alexnet':
            convNet = models.alexnet(pretrained=True)
            convNet.features = torch.nn.DataParallel(convNet.features)
    else:
        convNet = ConvNet().cuda()

    convNet = convNet.cuda()
    
    # define a cross entropy loss
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(convNet.parameters(), lr=0.001, momentum=0.9)
    
    # feed forward to test if the network is correctly setup
    temp = convNet(Variable(images.cuda()))

    # %% start train
    best_dev_acc = 0
    for epoch in range(20):
        running_loss = 0.0
        train_batches = 65
        curtime = time.time()
    
        # evaluation records
        correct = 0
        total = 0
    
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            
            if i <= train_batches:
                convNet.train()
                
                # wrap them in Variable
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward + backward + optimize
                outputs = convNet(inputs)
    
                loss = criterion(outputs, labels)
                loss.backward()        
                optimizer.step()
        
                running_loss += loss.data[0]
                
            else:
                convNet.eval()
                
                # wrap them in Variable
                inputs, labels = Variable(inputs.cuda()), labels.cuda()
                
                outputs = convNet(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                
        print('Epoch %d, Time elapse: %.2f, Train loss: %.5f Evaluation Accuracy: %d %%' % 
                  (epoch+1, time.time()-curtime,
                   running_loss / train_batches,
                   100 * float(correct) / total))
        
        # save the current model if better performance on validation set found
        if (float(correct) / total) > best_dev_acc:
            best_dev_acc = (correct / total)
            with open('model.ckpt', 'w') as f:
                torch.save(convNet.state_dict(), f)
        
    # %% evaluate on test set
    correct = 0
    total = 0
    
    for i, data in enumerate(testloader, 0):
        convNet.eval()
        
        # get the inputs
        inputs, labels = data
        
        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), labels.cuda()
        
        outputs = convNet(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        
    print('Test Accuracy: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    main()