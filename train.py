from data_onehot import Bandit_onehot
from data_multi import Bandit_multi
from data_sanity import Bandit_sanity
from learner_linear import LinearTS
from learner_neural import NeuralTS
from learner_diag import NeuralTSDiag
from learner_kernel import KernelTS
import numpy as np
import argparse
import pickle 
import os
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Thompson Sampling')
    parser.add_argument('--encoding', default='multi', metavar='sanity|onehot|multi')

    parser.add_argument('--dim', default=100, type=int, help='dim for linear bandit, sanity only')
    parser.add_argument('--arm', default=10, type=int, help='arm for linear bandit, sanity only')
    parser.add_argument('--noise', default=1, type=float, help='noise for linear bandit, sanity only')
    parser.add_argument('--size', default=1000, type=int, help='bandit size, sanity only')

    parser.add_argument('--dataset', default='mushroom', metavar='DATASET', help='encoding = onehot and multi only')
    parser.add_argument('--shuffle', type=bool, default=1, metavar='1 / 0', help='shuffle the data set or not, encoding = onehot and multi only')
    parser.add_argument('--seed', type=int, default=0, help='random seed for shuffle, 0 for None, encoding = onehot and multi only')

    parser.add_argument('--r', type=float, default=0.9, metavar='r', help='ratio for feature norm')

    parser.add_argument('--learner', default='linear', metavar='linear|neural|diag|kernel', help='TS learner')
    parser.add_argument('--style', default='ts', metavar='ts|ucb', help='TS or UCB')

    parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
    parser.add_argument('--lamdba', type=float, default=0.001, metavar='l', help='lambda for regularzation')    
    parser.add_argument('--hidden', type=int, default=100, help='network hidden size, learner = neural and diag only')
    parser.add_argument('--gamma', type=float, default=0.01, help='Gamma for RBF kernel, learner = kernel only')
    
    args = parser.parse_args()
    use_seed = None if args.seed == 0 else args.seed
    if args.encoding == 'sanity':
        b = Bandit_sanity(args.dim, args.noise, args.arm, args.size)
        bandit_info = 'sanity_{}_{}_{}_{}'.format(args.dim, args.noise, args.arm, args.size)
    elif args.encoding == 'onehot':
        b = Bandit_onehot(args.dataset, is_shuffle=args.shuffle, seed=use_seed, r=args.r)
        bandit_info = 'onehot_{}_{}'.format(args.dataset, args.r)
    elif args.encoding == 'multi':
        b = Bandit_multi(args.dataset, is_shuffle=args.shuffle, seed=use_seed)
        bandit_info = 'multi_{}'.format(args.dataset)
    else:
        raise RuntimeError('Encoding not exist')

    if args.learner == 'linear':
        l = LinearTS(b.dim, args.lamdba, args.nu, args.style)
        ts_info = '{}_linear_{:.3e}_{:.3e}'.format(args.style, args.lamdba, args.nu)
    elif args.learner == 'neural':
        l = NeuralTS(b.dim, args.lamdba, args.nu, args.hidden, args.style)
        ts_info = '{}_neural_{:.3e}_{:.3e}_{}'.format(args.style, args.lamdba, args.nu, args.hidden)
    elif args.learner == 'diag':
        l = NeuralTSDiag(b.dim, args.lamdba, args.nu, args.hidden, args.style)
        ts_info = '{}_diag_{:.3e}_{:.3e}_{}'.format(args.style, args.lamdba, args.nu, args.hidden)
    elif args.learner == 'kernel':
        l = KernelTS(b.dim, args.lamdba, args.nu, args.gamma, args.style)
        ts_info = '{}_kernel_{:.3e}_{:.3e}_{:.3e}'.format(args.style, args.lamdba, args.nu, args.gamma)

    regrets = []
    for t in range(b.size):
        context, rwd = b.step()
        arm_select, nrm, sig, ave_rwd = l.select(context)
        r = rwd[arm_select]
        reg = np.max(rwd) - r
        loss = l.train(context[arm_select], r)
        regrets.append(reg)
        print('{}: {:.3f}, {:.3e}, {:.3e}, {:.3e}, {:.3e}'.format(t, np.mean(regrets), loss, nrm, sig, ave_rwd))

    filename = '{:.3f}_{}_{}.pkl'.format(np.mean(regrets), bandit_info, ts_info)
    with open(os.path.join('record', filename), 'wb') as f:
        pickle.dump(regrets, f)
