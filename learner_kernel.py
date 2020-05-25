import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import rbf_kernel
import torch

class KernelTS:
    def __init__(self, dim, lamdba=1, nu=1, gamma=1, style='ts'):
        self.dim = dim
        self.lamdba = lamdba
        self.nu = nu
        self.gamma=gamma
        self.history_context = []
        self.history_reward = []
        self.history_len = 0
        self.scale = np.sqrt(self.lamdba * self.nu)
        self.style = style
    
    def select(self, context):
        a, f = context.shape
        if self.history_len == 0:
            mu = np.zeros((a,))
            sigma = self.scale * np.ones((a,))
        else:
            X_history = np.array(self.history_context)
            R_history = np.array(self.history_reward)
            if self.history_len >= 1000:
                # Calculate K_t, k_t
                K_t = torch.from_numpy(rbf_kernel(X_history, X_history, gamma=self.gamma)).cuda()
                k_t = torch.from_numpy(rbf_kernel(context, X_history, gamma=self.gamma)).cuda()
                r_t = torch.from_numpy(R_history).cuda()
                # (K_t + \lambda I)^{-1}
                U_t = torch.inverse(K_t + self.lamdba * torch.eye(self.history_len, device=torch.device('cuda'))) 
                mu_t = k_t.matmul(U_t.matmul(r_t))
                sigma_t = torch.diag(torch.ones((a,), device=torch.device('cuda')) - k_t.matmul(U_t.matmul(k_t.T)))
                mu = mu_t.cpu().numpy()
                sigma = sigma_t.cpu().numpy() * self.scale
            else:
                K_t = rbf_kernel(X_history, X_history, gamma=self.gamma)
                k_t = rbf_kernel(context, X_history, gamma=self.gamma)
                zz , _ = sp.linalg.lapack.dpotrf((self.lamdba * np.eye(self.history_len) + K_t), False, False)
                Linv, _ = sp.linalg.lapack.dpotri(zz)
                U_t = np.triu(Linv) + np.triu(Linv, k=1).T
                mu = np.dot(k_t, np.dot(U_t, R_history))
                sigma = self.scale * (np.ones((a,)) - np.diag(np.matmul(k_t, np.matmul(U_t, k_t.T))))

        if self.style == 'ts':
            r = np.random.multivariate_normal(mu, sigma)
        elif self.style == 'ucb':
            r = mu + sigma
        return np.argmax(r), 1, np.mean(sigma), np.mean(r)

    def train(self, context, reward):
        self.history_context.append(context)
        self.history_reward.append(reward)
        self.history_len += 1
        return 0
