import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim

class LinearTS:
    # Brute-force Linear TS with full inverse
    def __init__(self, dim, lamdba=1, nu=1):
        self.dim = dim
        self.U = lamdba * np.eye(dim)
        self.Uinv = 1 / lamdba * np.eye(dim)
        self.nu = nu
        self.jr = np.zeros((dim, ))
        self.mu = np.zeros((dim, ))

    def select(self, context):
        # theta = np.random.multivariate_normal(self.mu, self.nu * self.nu * self.Uinv)
        theta = self.mu
        return np.argmax(np.dot(context, theta))
    
    def train(self, context, reward):
        self.jr += reward * context
        self.U += np.matmul(context.reshape((-1, 1)), context.reshape((1, -1)))
        # fast inverse for symmetric matrix
        zz , _ = sp.linalg.lapack.dpotrf(self.U, False, False)
        Linv, _ = sp.linalg.lapack.dpotri(zz)
        self.Uinv = np.triu(Linv) + np.triu(Linv, k=1).T
        self.mu = np.dot(self.Uinv, self.jr)


class Network(nn.Module):
    def __init__(self, dim, hidden_size=1000):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

class NeuralTS:
    def __init__(self, dim, lamdba=1, nu=1):
        self.func = Network(dim, hidden_size=1000).cuda()
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba

    def select(self, context):
        tensor = torch.from_numpy(context).float().cuda()
        return torch.argmax(self.func(tensor)).item()
    
    def train(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)
        optimizer = optim.Adam(self.func.parameters(), weight_decay=self.lamdba)
        for _ in range(100):
            tot_loss = 0
            for c, r in zip(self.context_list, self.reward):
                optimizer.zero_grad()
                delta = self.func(c.cuda()) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()
        return tot_loss / len(self.reward)


