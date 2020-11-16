from data_onehot import Bandit_onehot
from data_multi import Bandit_multi
from data_sanity import Bandit_sanity
from learner_linear import LinearTS
from learner_neural import NeuralTS
from learner_diag import NeuralTSDiag
from learner_kernel import KernelTS
from neural_boost import Boost
from learner_diag_kernel import KernelTSDiag
from learner_diag_linear import LinearTSDiag
import numpy as np
import argparse
import pickle 
import os
import time
import torch


if __name__ == '__main__':
    t1 = time.time()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    parser = argparse.ArgumentParser(description='Thompson Sampling')
    parser.add_argument('--encoding', default='multi', metavar='sanity|onehot|multi')

    parser.add_argument('--dim', default=100, type=int, help='dim for linear bandit, sanity only')
    parser.add_argument('--arm', default=10, type=int, help='arm for linear bandit, sanity only')
    parser.add_argument('--noise', default=1, type=float, help='noise for linear bandit, sanity only')
    parser.add_argument('--size', default=10000, type=int, help='bandit size')

    parser.add_argument('--dataset', default='mushroom', metavar='DATASET', help='encoding = onehot and multi only')
    parser.add_argument('--shuffle', type=bool, default=1, metavar='1 / 0', help='shuffle the data set or not, encoding = onehot and multi only')
    parser.add_argument('--seed', type=int, default=0, help='random seed for shuffle, 0 for None, encoding = onehot and multi only')

    parser.add_argument('--r', type=float, default=0.9, metavar='r', help='ratio for feature norm, encoding = onehot only')

    parser.add_argument('--learner', default='linear', metavar='linear|neural|kernel|boost', help='TS learner')
    parser.add_argument('--inv', default='diag', metavar='diag|full', help='inverse matrix method')
    parser.add_argument('--style', default='ts', metavar='ts|ucb', help='TS or UCB')

    parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
    parser.add_argument('--lamdba', type=float, default=0.001, metavar='l', help='lambda for regularzation')    
    parser.add_argument('--hidden', type=int, default=100, help='network hidden size, learner = neural and diag only')

    parser.add_argument('--p', type=float, default=0.8, help='p, learner = boost only')
    parser.add_argument('--q', type=int, default=5, help='q, learner = boost only')
    parser.add_argument('--delay', type=int, default=1, help='delay reward')
    
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
        if args.inv == 'diag':
            """ Linear TS diag is to use a cuda network """
            l = LinearTSDiag(b.dim, args.lamdba, args.nu, args.style)
        elif args.inv == 'full':
            l = LinearTS(b.dim, args.lamdba, args.nu, args.style)
        else:
            RuntimeError('Inverse method not exist')
        ts_info = '{}_linear_{:.3e}_{:.3e}_{}'.format(args.style, args.lamdba, args.nu, args.inv)
    elif args.learner == 'neural':
        if args.inv == 'diag':
            l = NeuralTSDiag(b.dim, args.lamdba, args.nu, args.hidden, args.style)
        elif args.inv == 'full':
            l = NeuralTS(b.dim, args.lamdba, args.nu, args.hidden, args.style)
        else:
            RuntimeError('Inverse method not exist')
        ts_info = '{}_neural_{:.3e}_{:.3e}_{}_{}'.format(args.style, args.lamdba, args.nu, args.hidden, args.inv)
    elif args.learner == 'kernel':
        if args.inv == 'diag':
            raise RuntimeError('Diag inverse estimation not feasible to kernel method!')
            l = KernelTSDiag(b.dim, args.lamdba, args.nu, args.style)
        elif args.inv == 'full':
           l = KernelTS(b.dim, args.lamdba, args.nu, args.style)
        else:
            RuntimeError('Inverse method not exist')
        ts_info = '{}_kernel_{:.3e}_{:.3e}_{}'.format(args.style, args.lamdba, args.nu, args.inv)
    elif args.learner == 'boost':
        l = Boost(b.dim, args.hidden, args.p, args.q)
        ts_info = 'boost_{:.1e}_{}_{}'.format(args.p, args.q, args.hidden)
    else:
        raise RuntimeError('Learner not exist')
    setattr(l, 'delay', args.delay)

    regrets = []
    for t in range(min(args.size, b.size)):
        context, rwd = b.step()
        arm_select, nrm, sig, ave_rwd = l.select(context)
        r = rwd[arm_select]
        reg = np.max(rwd) - r
        loss = l.train(context[arm_select], r)
        regrets.append(reg)
        if t % 100 == 0:
            print('{}: {:.3f}, {:.3e}, {:.3e}, {:.3e}, {:.3e}'.format(t, np.sum(regrets), loss, nrm, sig, ave_rwd))

    filename = '{:.3f}_{}_{}_delay_{}_{}.pkl'.format(np.sum(regrets), bandit_info, ts_info, args.delay, time.time() - t1)
    with open(os.path.join('record', filename), 'wb') as f:
        pickle.dump(regrets, f)
