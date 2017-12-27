import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import floor

import IPython as ipy

class LinearActorCritic(nn.Module):
    def __init__(self, H_in=100, nc=2, na=4):
        super(LinearActorCritic, self).__init__()
        self.H_in = H_in
        self.nc = nc
        self.na = na

        self.policy_linear = nn.Linear(self.nc*self.H_in*self.H_in, 4)
        self.critic_linear = nn.Linear(self.nc*self.H_in*self.H_in, 1)

    def forward(self, x):
        x = x.contiguous().view(-1, self.nc*self.H_in*self.H_in)

        pa = F.softmax(self.policy_linear(x))
        V = self.critic_linear(x)

        return pa, V.view(-1)

class MLPActorCritic(nn.Module):
    def __init__(self, H_in=100, nc=2, na=4):
        super(MLPActorCritic, self).__init__()
        self.H_in = H_in
        self.nc = nc
        self.na = na

        self.linear = nn.Linear(self.nc*self.H_in*self.H_in, 256)
        self.policy_linear = nn.Linear(256, self.na)
        self.critic_linear = nn.Linear(256, 1)

    def forward(self, x):
        x = x.contiguous().view(-1, self.nc*self.H_in*self.H_in)

        x = self.linear(x)
        x = F.relu(x)
        pa = F.softmax(self.policy_linear(x))
        V = self.critic_linear(x)

        return pa, V.view(-1)

class CNNActorCritic(nn.Module):
    def __init__(self, H_in=100, nc=2, na=4):
        super(CNNActorCritic, self).__init__()
        self.H_in = H_in
        self.nc = nc
        self.na = na

        self.conv1 = nn.Conv2d(self.nc, 32, 8, stride=2)
        self.H_1 = floor((self.H_in - (8-1)-1)/2+1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.H_2 = floor((self.H_1 - (4-1)-1)/2+1)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.H_3 = floor((self.H_2 - (3-1)-1)/1+1)
        assert self.H_3 > 0
        print (self.H_in, "x", self.H_in, "-> %d" % (self.H_3*self.H_3*32))

        self.linear1 = nn.Linear(32 * self.H_3 * self.H_3 + 5*5, 64)
        self.policy_linear = nn.Linear(64, self.na)
        self.softmax = nn.Softmax()
        self.critic_linear = nn.Linear(64, 1)

    def forward(self, x):
        mid = self.H_in // 2
        immediate = x[:,0,mid-2:mid+3,mid-2:mid+3].contiguous()
        immediate = immediate.view(immediate.size()[0], 5*5)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * self.H_3 * self.H_3)
        x = self.linear1(torch.cat([x, immediate], dim=1))
        x = F.relu(x)

        pa = self.softmax(self.policy_linear(x))
        v = self.critic_linear(x)

        return pa, v.view(-1)

class ResidualBlock(nn.Module):
    def __init__(self, nc):
        super(ResidualBlock, self).__init__()

        self.nc = nc
        self.conv1 = nn.Conv2d(self.nc, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        out = F.relu(out)

        return out

class ResNetActorCritic(nn.Module):
    def __init__(self, H_in=100, nc=2, na=4):
        super(ResNetActorCritic, self).__init__()
        self.H_in = H_in
        self.nc = nc
        self.na = na

        self.conv1 = nn.Conv2d(self.nc, 32, kernel_size=3, stride=1, padding=1)
        self.tower = nn.Sequential(*[ResidualBlock(32) for _ in range(6)])

        self.linear = nn.Linear(32*self.H_in*self.H_in, 256)
        self.policy_linear = nn.Linear(256, self.na)
        self.critic_linear = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.tower(x)
        x = x.view(-1, 32*self.H_in*self.H_in)

        x = self.linear(x)
        x = F.relu(x)
        pa = F.softmax(self.policy_linear(x))
        V = self.critic_linear(x)

        return pa, V.view(-1)

if __name__ == '__main__':
    obs = torch.randn(32, 2, 70, 70)
    obsv = Variable(obs)

    pi = CNNActorCritic(H_in = 70, nc = 2, na = 4)
    pa, v = pi(obsv)