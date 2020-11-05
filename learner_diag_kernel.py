import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import rbf_kernel
import torch

class KernelTSDiag:
    def __init__(self, dim, lamdba=1, nu=1, style='ts'):
        self.dim = dim
        self.lamdba = lamdba
        self.nu = nu
        self.history_context = []
        self.ker_diag = []
        self.history_reward = []
        self.history_len = 0
        self.scale = np.sqrt(self.lamdba * self.nu)
        self.style = style
    
    def select(self, context):
        a, _ = context.shape
        if self.history_len == 0:
            mu = np.zeros((a,))
            sigma = self.scale * np.ones((a,))
        else:
            X_history = np.array(self.history_context)
            R_history = np.array(self.history_reward)

            k_t = rbf_kernel(context, X_history)
            U_t = 1 / (self.lamdba * np.ones((self.history_len)) + np.array(self.ker_diag))
            mu = np.dot(k_t, U_t * R_history)
            sigma = np.zeros((a,))
            for i in range(a):
                print(k_t[i].shape)
                sigma[i] = self.scale * np.dot(k_t[i], U_t * k_t[i])
            

        if self.style == 'ts':
            r = np.random.multivariate_normal(mu, np.diag(sigma * sigma))
        elif self.style == 'ucb':
            r = mu + sigma
        return np.argmax(r), 1, np.mean(sigma), np.mean(r)

    def train(self, context, reward):
        self.history_context.append(context)
        self.history_reward.append(reward)
        self.ker_diag.append(rbf_kernel(context.reshape((1, -1)), context.reshape((1, -1)))[0, 0])
        
        self.history_len += 1
        return 0
