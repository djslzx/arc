"""
Perceptual loss

- [x] Generate random expressions in DSL
  - pick a grammar element at random, then recursively pick subcomponents if nonterminal
- [x] Generate random choices of Z
- Use Pytorch to train NN to measure distance in f-space:
  - convolve each example x_i to get a vector
  - use max pooling to combine multiple x_i's into a vector
  - train NN to learn that two vectors are similar if the f's that generated them are similar
    - extension: use some notion of distance between expressions: edit distance?
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import itertools
import pickle

from grammar import *
from bottom_up import *
from learn import *


class Net(nn.Module):

    def __init__(self, n, c=6, w=B_W, h=B_H):
        """
        n: number of elts in each example set
        """
        super(Net, self).__init__()
        self.n = n
        self.c = c
        self.w = w
        self.h = h
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, c, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, padding='same'),
            nn.ReLU(),
        )
        k = 2 * c * w * h
        self.linear_stack = nn.Sequential(
            nn.Linear(k, k),
            nn.ReLU(),
            nn.Linear(k, k),
            nn.ReLU(),
            nn.Linear(k, k),
            nn.ReLU(),
            nn.Linear(k, 1),
        )

    def forward(self, x):
        """
        x: a tensor w/ shape (2N x 1 x Bw x Bh)
        """
        x = x.squeeze(dim=0)
        x = self.conv_stack(x)

        # Split, max pool, flatten, and concat
        a, b = x.split(self.n)
        a = T.max(a, dim=0).values
        b = T.max(b, dim=0).values
        x = T.cat((a, b))

        x = x.flatten()
        x = self.linear_stack(x)
        # x = x.unsqueeze(0)
        return x

    def train(self, xs, ys, epochs):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = T.optim.Adam(net.parameters(), lr=0.001)

        dataset = T.utils.data.TensorDataset(xs, ys)
        dataloader = T.utils.data.DataLoader(dataset, shuffle=True)

        for epoch in range(1, epochs+1):
            for i, (x,y) in enumerate(dataloader, 1):
                optimizer.zero_grad()
                out = self(x)
                print("out:", out, "y:", y)
                loss = criterion(out, y.squeeze(0))
                loss.backward()
                optimizer.step()

            # print statistics
            print('[%d] loss: %.3f' % (epoch, loss.item()))

        T.save(self.state_dict(), './model.pt')
        print('Finished Training')

def gen_exprs(grammar, zs, return_type, max_size):
    envs = [{'z_n':z} for z in zs]
    return list(p for p, _ in bottom_up_generator(max_size, grammar, envs)
                if p.return_type == return_type)

def percent_empty(xs):
    def empty(t): return t.sum() == 0
    return sum(empty(x) for x in xs)/len(xs)

def test_percent_empty():
    tests = [
        ([T.ones(4,4)], 0),
        ([T.zeros(4,4)], 1),
        ([T.zeros(4,4), T.zeros(4,4)], 1),
        ([T.zeros(4,4), T.zeros(4,4), T.ones(4,4)], 0.66),
        ([T.zeros(4,4), T.zeros(4,4), T.ones(4,4), T.ones(4,4)], 0.5),
    ]
    for args, expected in tests:
        actual = percent_empty(args)
        assert abs(actual - expected) < 0.01, \
            f"test failed: percent_empty({args})={actual}, expected={expected}"

def make_exs(exprs, zs, n_fs, k_f, n_xs, empty_thresh=0.3):
    """
    Generate exs = [{f(z)} | f is an expr]

    n_fs: number of unique f's to use
    k_f: number of times to repeat each f
    n_xs: number of xs to generate for each example
    """
    def eval(f, z):
        try:
            return f.eval({'z_n': z}).as_tensor()
        except AssertionError:
            return T.zeros(B_W, B_H)

    pools = {}
    for _ in range(n_fs):
        # Choose an f from fs that has at least half of its outputs nonempty on n_xs * k_f examples
        while True:
            f = random.choice(exprs)
            pool = [eval(f, z) for z in random.sample(zs, n_xs * k_f)]
            if percent_empty(pool) < empty_thresh: break
        pools[f] = pool

    ins, outs = [], []
    for f, f_pool in pools.items():
        for _ in range(k_f):
            # Add matching examples using xs generated with f
            ins.append(T.stack(random.sample(f_pool, n_xs) + 
                               random.sample(f_pool, n_xs)))
            outs.append(1.0)

            # Add non-matching examples by pairing f with g
            assert n_fs > 1, "Tried to generate non-matching f's when there's only one f"
            while True:
                g, g_pool = random.choice(list(pools.items()))
                if g != f: break
            f_xs = random.sample(f_pool, n_xs)
            g_xs = []
            while True:
                g_xs = random.sample(g_pool, n_xs)
                if T.abs(T.sum(T.stack(g_xs) - T.stack(f_xs))) < 0.001: break
            ins.append(T.stack(f_xs + g_xs))
            outs.append(0.0)

    return ins, outs

if __name__ == '__main__':
    # test_percent_empty()

    g = Grammar(ops=[Plus, Times, Minus, Rect, Program],
                consts=[Zn(Num(i)) for i in range(Z_SIZE)] + [Num(i) for i in range(Z_LO, Z_HI + 1)])
    n_fs = 2
    k_f = 1
    n_xs = 1
    n_zs = 1000
    epochs = 10000
    depth = 5
    print(f"Params: depth={depth}, n_fs={n_fs}, k_f={k_f}, n_xs={n_xs}, n_zs={n_zs}")

    # print('Making and writing Z...')
    # zs = [gen_zn() for _ in range(n_zs)]
    # with open('../data/zs.dat', 'wb') as f:
    #     pickle.dump(zs, f)

    # print('Making and writing exprs...')
    # exprs = gen_exprs(g, zs, 'Bitmap', depth)
    # with open('../data/exprs.dat', 'wb') as f:
    #     pickle.dump(exprs, f)

    print('Making examples...')
    # load exprs, zs
    exprs = []
    zs = []
    with open('../data/exprs.dat', 'rb') as f:
        exprs = pickle.load(f)
    with open('../data/zs.dat', 'rb') as f:
        zs = pickle.load(f)
    # make exs
    xs, ys = make_exs(exprs, zs, n_fs, k_f, n_xs)
    # save exs
    print('Saving examples...')
    with open('../data/exs.dat', 'wb') as f:
        pickle.dump((xs, ys), f)

    # print('Loading exs from file...', end=' ')
    # xs, ys = None, None
    # with open('../data/exs.dat', 'rb') as f:
    #     xs, ys = pickle.load(f)

    # Format examples
    n_exs = len(xs)
    xs = T.stack([T.reshape(x, (n_xs * 2, 1, B_W, B_H)) for x in xs])
    ys = T.reshape(T.tensor(ys), (n_exs, 1)) # reshape ys into a column vector
    print(xs.shape, ys.shape)
    threshold = 0.5

    # Build and train NN
    print('Training NN...')
    net = Net(n=n_xs)
    net.train(xs, ys, epochs)

    # Test NN
    print('Testing NN...')
    n_correct = 0
    fps, fns, tps, tns = 0, 0, 0, 0
    for i, (x, y) in enumerate(zip(xs, ys)):
        y = bool(y.item())
        prediction = net(x).item() > threshold

        tps += y and prediction
        tns += not y and not prediction
        fps += not y and prediction
        fns += y and not prediction

        n_correct += y == prediction

    print("|correct|:", n_correct, "|exs|:", n_exs, "correct %:", n_correct/n_exs * 100)
    print("fps:", fps, "fns:", fns)

    # for f, xs in exs:
    #     print('f:', f.pretty_print())
    #     for x in xs:
    #         for e in x:
    #             print(e.pretty_print())
    #             print()
    #         print()
