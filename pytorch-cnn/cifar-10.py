from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool  = nn.MaxPool2d(2,2)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def main():
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1]
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
    trainset = torchvision.datasets.CIFAR10(root='./cifar10_dataset', 
                                            train=True, download=True, 
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=20,
                                              shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./cifar10_dataset', 
                                           train=False, download=True, 
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=20,
                                             shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # show some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    # print images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    # initialize the network    
    net = Net()
    net = net.cuda()
    
    criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(20): # loop over the dataset multiple times
        running_loss = 0.0
        curtime = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            
            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()        
            optimizer.step()
            
            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999: # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f - elapsed time: %.2f' % 
                      (epoch+1, i+1, running_loss / 2000, time.time()-curtime))
                running_loss = 0.0
                curtime = time.time()
    
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = Variable(images).cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    
    print('Finished Training')

if __name__ == '__main__':
    main()