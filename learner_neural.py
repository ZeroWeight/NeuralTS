import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim

class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))
        
class NeuralTS:
    def __init__(self, dim, lamdba=1, nu=1, hidden=100, style='ts'):
        self.func = Network(dim, hidden_size=hidden).cuda()
        self.func1 = Network(dim, hidden_size=hidden).cuda()
        self.func1.load_state_dict(self.func.state_dict())
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,)).cuda()
        self.nu = nu
        self.style = style

    def select(self, context):
        tensor = torch.from_numpy(context).float().cuda()
        mu = self.func(tensor)

        mu1 = self.func1(tensor)
        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        for fx in mu1:
            self.func1.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func1.parameters()])
            g_list.append(g)
            sigma2 = self.lamdba * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            if self.style == 'ts':
                sample_r = np.random.normal(loc=fx.item(), scale=sigma.item())
            elif self.style == 'ucb':
                sample_r = fx.item() + sigma.item()
            else:
                raise RuntimeError('Exploration style not set')
            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r
        arm = np.argmax(sampled)
        self.U += g_list[arm] * g_list[arm]
        return arm, g_list[arm].norm().item(), ave_sigma, ave_rew
    
    def train(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)
        optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                optimizer.zero_grad()
                delta = self.func(c.cuda()) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length
