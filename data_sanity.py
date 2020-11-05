import numpy as np

class Bandit_sanity:
    def __init__(self, dim, noise, arms, size):
        self.dim = dim
        self.theta = np.random.uniform(low=-1, high=1, size=(dim,))
        self.arms = arms
        self.noise = noise * np.eye(self.arms)
        self.size = size
    
    def step(self):
        x = np.random.uniform(low=-1, high=1, size=(self.arms, self.dim))
        r = np.dot(x, self.theta)
        r_noise = np.random.multivariate_normal(r, self.noise)
        return x, r

    
if __name__ == '__main__':
    b = Bandit_sanity(100, 1, 3)
    b.step()