import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as f
from torch.distributions.normal import Normal

class Network(nn.Module):
    def __init__(self, dim, m=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, m, bias=False)
        self.fc1a = nn.Linear(dim, m, bias=False)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(m, 1, bias=False)
        self.fc2a = nn.Linear(m, 1, bias=False)
        self.scale = math.sqrt(m)
        
        layer_std = math.sqrt(2) / self.scale
        nn.init.normal_(self.fc1.weight, mean=0.0, std=layer_std)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=layer_std)
        nn.init.zeros_(self.fc1a.weight)
        nn.init.zeros_(self.fc2a.weight)

        self.fc1.weight.requires_grad = False
        self.fc2.weight.requires_grad = False
    def forward(self, x):
        hidden = self.fc1(x) + self.fc1a(x)
        _x = self.activate(hidden)
        output = self.fc2(_x) + self.fc2a(_x)
        return output * self.scale
    
class NeuralTSDiag:
    def __init__(self, dim, lamdba=1, nu=1, hidden=100, style='ts'):
        self.func = Network(dim, m=hidden).cuda()
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,)).cuda()
        self.nu = nu
        self.style = style
        self.m = hidden

    def select(self, context):
        x = torch.tensor(context, dtype=torch.float, device=torch.device('cuda'))
        mu = self.func(x)
        g_list = []
        sampled = []
        for fx in mu:
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() if p.requires_grad else torch.tensor([], device=torch.device('cuda'))
             for p in self.func.parameters()])
            g_list.append(g)
            sigma2 = torch.sum(self.lamdba * self.nu * self.nu * g * g / self.U)
            sigma = torch.sqrt(sigma2)
            if self.style == 'ucb':
                sampled.append(fx + sigma)
            elif self.style == 'ts':
                sampled.append(Normal(fx, sigma))
        return np.argmax(sampled), 0, 0, 0

    def train(self, context, reward):
        self.context_list.append(context)
        self.reward.append(reward)
        length = len(self.reward)
        optimizer = optim.SGD(self.func.parameters(), lr=1e-3, weight_decay=1)
        C = torch.tensor(self.context_list, dtype=torch.float, device=torch.device('cuda'))
        R = torch.tensor(self.reward, dtype=torch.float, device=torch.device('cuda'))
        train_len = 100 if length % 10 == 1 else 10
        for _ in range(train_len):
            Y = self.func(C).view(-1)
            optimizer.zero_grad()
            loss = 0.5 * f.mse_loss(Y, R, reduction='sum') / (self.lamdba * self.m)
            loss.backward()
            optimizer.step()
        return loss.item()

