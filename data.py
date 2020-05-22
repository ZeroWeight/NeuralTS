from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np

class Bandit:
    def __init__(self, name, is_shuffle=True, seed=None, r=0.5):
        # Fetch data
        if name == 'mnist':
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        elif name == 'mushroom':
            X, y = fetch_openml('mushroom', version=1, return_X_y=True)
        elif name == 'adult':
            X, y = fetch_openml('adult', version=2, return_X_y=True)
        elif name == 'covertype':
            X, y = fetch_openml('covertype', version=3, return_X_y=True)
        elif name == 'isolet':
            X, y = fetch_openml('isolet', version=1, return_X_y=True)
        elif name == 'letter':
            X, y = fetch_openml('letter', version=1, return_X_y=True)
        elif name == 'MagicTelescope':
            X, y = fetch_openml('MagicTelescope', version=1, return_X_y=True)
        elif name == 'shuttle':
            X, y = fetch_openml('shuttle', version=1, return_X_y=True)
        else:
            raise RuntimeError('Dataset does not exist')
        # avoid nan, set nan as -1
        X[np.isnan(X)] = np.min(X) - 1
        X = np.sqrt(r) * normalize(X)
        # Shuffle data
        if is_shuffle:
            self.X, self.y = shuffle(X, y, random_state=seed)
        else:
            self.X, self.y = X, y
        # generate one_hot coding:
        self.y_arm = OrdinalEncoder(dtype=np.int).fit_transform(self.y.reshape((-1, 1)))
        # cursor and other variables
        self.cursor = 0
        self.size = self.y.shape[0]
        self.n_arm = np.max(self.y_arm) + 1
        self.dim = self.X.shape[1] + self.n_arm
        self.eye = np.sqrt(1 - r) * np.eye(self.n_arm)

    def step(self):
        assert self.cursor < self.size
        x = self.X[self.cursor].reshape((1, -1)).repeat(self.n_arm, axis=0)
        x_y = np.hstack((x, self.eye))
        arm = self.y_arm[self.cursor][0]
        self.cursor += 1
        return x_y, arm

    def finish(self):
        return self.cursor == self.size
    def reset(self):
        self.cursor = 0

if __name__ == '__main__':
    b = Bandit('mnist', r=0.3)
    x_y, a = b.step()
    print(np.linalg.norm(x_y, axis=1))