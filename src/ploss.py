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

def is_recursive(t):
    return t.return_type in t.argument_types

def is_const(grammar_elt):
    """Check whether `grammar_elt` (an op or const) is a const"""
    return not grammar_elt.argument_types or type(grammar_elt) in [Zb, Zn]

def gen_exprs(grammar, zs, return_type, max_size):
    envs = [{'z_n':z} for z in zs]
    return list(p for p, _ in bottom_up_generator(max_size, grammar, envs)
                if p.return_type == return_type)

def make_exs(exprs, zs, n_fs, k_f, n_xs):
    """
    Generate exs = [{f(z)} | f is an expr]

    n_fs: number of unique f's to use
    k_f: number of times to repeat each f
    n_xs: number of xs to generate for each example
    """
    def eval(f, z):
        try:
            return f.eval({'z_n':z}).as_tensor().float()
        except AssertionError:
            return T.zeros(B_W, B_H)

    exs = []
    for _ in range(n_fs):
        sets = []
        # Choose an f from fs that has at least half of its outputs nonempty
        while True:
            f = random.choice(exprs)
            xs = [eval(f, z) for z in random.sample(zs, n_xs)]
            p_empty = sum(T.sum(x) == 0 for x in xs)/len(xs)
            if p_empty < 0.5: break

        # Generate more sets of xs using the same f
        # and turn each bitmap into a tensor
        sets = [xs] + [[eval(f,z) for z in random.sample(zs, n_xs)]
                       for _ in range(k_f - 1)]
        exs.append((f, [T.stack(s) for s in sets]))

    ins, outs = [], []
    for f, f_sets in exs:
        # Add matching examples using xs generated with f
        for pair in itertools.combinations(f_sets, 2):
            ins.append(T.cat(pair))
            outs.append(1.0)

        # Add non-matching examples by pairing f with g
        for _ in range(k_f):
            while True:
                g, g_sets = random.choice(exs)
                if g != f: break
            ins.append(T.cat((g_sets[0], f_sets[0])))
            outs.append(0.0)

    return ins, outs

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
            running_loss = 0.0
            for i, (x,y) in enumerate(dataloader, 1):
                # print(x[0][0])

                optimizer.zero_grad()
                loss = criterion(self(x), y.squeeze(0))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 300 == 0:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch, i, running_loss / 300))
                    running_loss = 0.0

        T.save(self.state_dict(), './model.pt')
        print('Finished Training')

if __name__ == '__main__':
    g = Grammar(ops=[Plus, Times, Minus, Rect, Program],
                consts=[Zn(Num(i)) for i in range(Z_SIZE)] + [Num(i) for i in range(Z_LO, Z_HI + 1)])
    n_fs = 100
    k_f = 3
    n_xs = 50
    n_zs = 1000
    print(f"Params: n_fs={n_fs}, k_f={k_f}, n_xs={n_xs}, n_zs={n_zs}")

    # print('Making and writing Z...')
    # zs = [gen_zn() for _ in range(n_zs)]
    # with open('../data/zs.dat', 'wb') as f:
    #     pickle.dump(zs, f)

    # print('Making and writing exprs...')
    # exprs = gen_exprs(g, zs, 'Bitmap', 5)
    # with open('../data/exprs.dat', 'wb') as f:
    #     pickle.dump(exprs, f)

    # print('Making and saving examples...')
    # exprs = []
    # zs = []
    # with open('../data/exprs.dat', 'rb') as f:
    #     exprs = pickle.load(f)
    # with open('../data/zs.dat', 'rb') as f:
    #     zs = pickle.load(f)
    # exs = make_exs(exprs, zs, n_fs, k_f, n_xs)
    # with open('../data/exs.dat', 'wb') as f:
    #     pickle.dump(exs, f)

    print('Loading exs from file...', end=' ')
    exs = None
    with open('../data/exs.dat', 'rb') as f:
        exs  = pickle.load(f)
        print('done.')

    # Format examples
    xs, ys = exs
    n_exs = len(xs)
    xs = T.stack([T.reshape(x, (n_xs * 2, 1, B_W, B_H)) for x in xs])
    ys = T.reshape(T.tensor(ys), (n_exs, 1)) # reshape ys into a column vector
    print(xs.shape, ys.shape)
    threshold = 0.5

    # Build and train NN
    net = Net(n=n_xs)
    net.train(xs, ys, epochs=1)

    # Test NN
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
