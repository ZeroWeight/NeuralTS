import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
import random

class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.context_list = []
        self.reward = []

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))
        
class Boost:
    def __init__(self, dim, hidden=100, p=0.8, q=5):
        self.p = p
        self.q = q
        self.func = [Network(dim, hidden_size=hidden).cuda() for i in range(q)]


    def select(self, context):
        arms, _ = context.shape
        if self.q == 1 and np.random.random()  < 0.05:
            return np.random.randint(arms), 0, 0, 0
        else:
            tensor = torch.from_numpy(context).float().cuda()
            mu = random.choice(self.func)(tensor).reshape((-1,))
            arm = torch.argmax(mu)
            return arm, 0, 0, 0
    
    def train(self, context, reward):
        ret = 0
        M = 10000
        for q in range(self.q):
            if random.random() < self.p:
                self.func[q].context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
                self.func[q].reward.append(reward)
            length = len(self.func[q].reward)
            if length == 0:
                break
            optimizer = optim.SGD(self.func[q].parameters(), lr=1e-2)
            index = np.arange(length)
            np.random.shuffle(index)
            cnt = 0
            tot_loss = 0
            while True:
                batch_loss = 0
                ex = False
                for idx in index:
                    c = self.func[q].context_list[idx]
                    r = self.func[q].reward[idx]
                    optimizer.zero_grad()
                    delta = self.func[q](c.cuda()) - r
                    loss = delta * delta
                    loss.backward()
                    optimizer.step()
                    batch_loss += loss.item()
                    tot_loss += loss.item()
                    cnt += 1
                    if cnt >= M:
                        ret += tot_loss / M
                        ex = True
                        break

                if batch_loss / length <= 1e-3:
                    ret += batch_loss / length
                    break
                if ex:
                    break
        return ret / self.q