import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channels=4, n_actions=4):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super().__init__()
        #self.bn0 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)
        # 4 84x84
        # 32 20x20
        # 64 9x9
        # 64 7x7


    def forward(self, x):
        #x = self.bn0(x.float())
        x = x.float()/255
        x = self.bn1(F.selu(self.conv1(x)))
        x = self.bn2(F.selu(self.conv2(x)))
        x = self.bn3(F.selu(self.conv3(x)))
        x = F.selu(self.fc4(x.reshape(x.size(0), -1)))
        return self.head(x)

class OCNN(nn.Module):
    def __init__(self, in_channels=4, n_actions=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)
    def forward(self, x):
        x = x.float()/255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(self.flatten(x)))
        return self.head(x)

class DCNN(nn.Module):
    def __init__(self, in_channels=4, n_actions=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(7 * 7 * 64, 512)
        self.head1 = nn.Linear(512, 1)
        self.head2 = nn.Linear(512, n_actions)
    def forward(self, x):
        x = x.float()/255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        a = self.head1(F.relu(self.fc1(x)))
        b = self.head2(F.relu(self.fc2(x)))
        return a+b-b.mean(1,True)


class SMM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(10,10, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(10,10, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc4 = nn.Linear(7 * 7 * 10, 1)
    def forward(self, x):
        x = x.float()/255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.fc4(self.flatten(x))

class FCC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        #self.bn1 = nn.BatchNorm2d(32)
        self.fc4 = nn.Linear(20 * 20 * 32, 16)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        return self.fc4(x.reshape(x.size(0), -1))

class FC(nn.Module):
    def __init__(self,*w):
        super().__init__()
        #self.bn = nn.BatchNorm2d(1)
        self.l = [nn.Linear(w[i],w[i+1]) for i in range(len(w)-1)]
        for i in range(len(w)-1):
            self.__setattr__('l%d'%i,self.l[i])
    def forward(self, x):
        x = (x/255.0).flatten()
        for i in self.l[:-1]:
            x = F.relu(i(x))
        return self.l[-1](x)