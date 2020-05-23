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