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

import util
from grammar import *
from bottom_up import *
from learn import *

# Use a GPU
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
if T.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")

# Tensorboard
import torch.utils.tensorboard as tb

class Net(nn.Module):

    def __init__(self, n, c=6):
        """
        n: number of elts in each example set
        """
        super(Net, self).__init__()
        self.n = n
        self.c = c
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, c, 3, padding='same'),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, padding='same'),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, padding='same'),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, padding='same'),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, padding='same'),
            nn.BatchNorm2d(c),
            nn.ReLU(),
        )
        k = 2 * c * B_H * B_W
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

    # def log_embeddings(self, writer, data, labels):

    #     def select_n_random(data, labels, n=100):
    #         assert len(data) == len(labels)
    #         perm = T.randperm(len(data))
    #         return data[perm][:n], labels[perm][:n]

    #     # select random images and their target indices
    #     data, labels = select_n_random(data, labels)

    #     # log embeddings
    #     features = data.view(40, -1)
    #     print(features.shape)
    #     writer.add_embedding(features, metadata=labels)

    def train(self, data, labels, epochs):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = T.optim.Adam(self.parameters(), lr=0.001)

        dataset = T.utils.data.TensorDataset(data, labels)
        dataloader = T.utils.data.DataLoader(dataset, shuffle=True)
        self.to(device)
        
        writer = tb.SummaryWriter() # tensorboard
        
        for epoch in range(1, epochs+1):
            for i, (x,y) in enumerate(dataloader, 1):
                optimizer.zero_grad()
                out = self(x)
                loss = criterion(out, y.squeeze(0))
                loss.backward()
                optimizer.step()

            # print statistics
            print('[%d/%d] loss: %.10f' % (epoch, epochs, loss.item()), end='\r')
            writer.add_scalar('training loss', loss.item(), epoch)

            if loss.item() == 0:
                print('[%d/%d] loss: %.3f' % (epoch, epochs, loss.item()))
                break

        T.save(self.state_dict(), './model.pt')
        print('Finished Training')

def gen_zs(n):
    zs = (T.rand(n, Z_SIZE) * (Z_HI - Z_LO) - Z_LO).long()
    return zs

def gen_exprs(grammar, zs, return_type, max_size):
    envs = [{'z_n': z} for z in zs]
    # for env in envs: print(" ", env)
    return list(p for p, _ in bottom_up_generator(max_size, grammar, envs)
                if p.return_type == return_type)

def empty(t): return t.sum() == 0

def percent_empty(xs):
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
    assert n_xs * k_f <= len(zs), "not enough zs to sample xs from"

    def eval(f, z):
        try:
            return f.eval({'z_n': z}).as_tensor()
        except AssertionError:
            return T.zeros(B_W, B_H)

    # print(type(zs), type(zs[0]), type(zs[0][0]))
    zs = set(zs)
    exprs = set(exprs)
    pools = {}
    seen = set()
    while len(pools) < n_fs:
        f = random.choice(list(exprs - pools.keys() - seen))
        seen.add(f)
        xs = []
        for z in zs:
            x = eval(f, z)
            if not empty(x):
                xs.append(x)
                if len(xs) >= n_xs * k_f: 
                    pools[f] = xs
                    break

    data, labels = [], []
    for f, f_pool in pools.items():
        # Add matching examples using xs generated with f
        for p1, p2 in util.chunk_pairs(f_pool, n_xs, k_f):
            v1 = T.stack(p1)
            v2 = T.stack(p2)
            data.append(T.cat((v1, v2)))
            labels.append(1.0)

        # Add non-matching examples by pairing f with some g
        assert n_fs > 1, "Tried to generate non-matching f's when there's only one f"
        non_matching = 0
        while non_matching < k_f:
            g = random.choice(list(pools.keys() - {f}))
            g_pool = pools[g]

            f_xs = T.stack(random.sample(f_pool, n_xs))
            g_xs = T.stack(random.sample(g_pool, n_xs))
            if not f_xs.isclose(g_xs).all():
                data.append(T.cat((f_xs, g_xs)))
                labels.append(0.0)
                non_matching += 1

    return data, labels

def make_zs_and_exprs():
    print('Making and writing Z...')
    zs = gen_zs(n_zs)
    with open('../data/zs.dat', 'wb') as f:
        pickle.dump(zs, f)

    print('Making and writing exprs...')
    exprs = gen_exprs(g, zs, 'Bitmap', depth)
    with open('../data/exprs.dat', 'wb') as f:
        pickle.dump(exprs, f)
    return zs, exprs

def load_zs_and_exprs():
    print('Loading Z, F...')
    with open('../data/exprs.dat', 'rb') as f: 
        exprs = pickle.load(f)
    with open('../data/zs.dat', 'rb') as f: 
        zs = pickle.load(f)
    return zs, exprs
    
def make_and_store_exs(zs, exprs, n_fs, k_f, n_xs):
    print('Making examples...')
    xs, ys = make_exs(exprs, zs, n_fs, k_f, n_xs)

    print('Saving examples...')
    with open('../data/exs.dat', 'wb') as f:
        pickle.dump((xs, ys), f)
    return xs, ys

def load_exs():
    print('Loading exs from file...', end=' ')
    xs, ys = None, None
    with open('../data/exs.dat', 'rb') as f:
        xs, ys = pickle.load(f)
    return xs, ys

def train_nn(net, n_xs, xs, ys, epochs):
    print('Training NN...')
    net.to(device)
    net.train(xs, ys, epochs)

def test_nn(net, xs, ys, threshold):
    print('Testing NN...')
    n_correct = 0
    fps, fns, tps, tns = 0, 0, 0, 0
    for i, (x, y) in enumerate(zip(xs, ys)):
        y = bool(y.item())
        prediction = net(x).item() > threshold
        fps += not y and prediction
        fns += y and not prediction
        n_correct += y == prediction

    print("|correct|:", n_correct, "|exs|:", n_exs, "correct %:", n_correct/n_exs * 100)
    print("fps:", fps, "fns:", fns)

if __name__ == '__main__':
    # test_percent_empty()

    g = Grammar(ops=[Plus, Times, Minus, Rect, Program],
                consts=[Zn(Num(i)) for i in range(Z_SIZE)] + [Num(i) for i in range(Z_LO, Z_HI + 1)])
    n_fs = 100
    k_f = 2
    n_xs = 5
    n_zs = 1000
    epochs = 10000
    depth = 5
    threshold = 0.5
    print(f"Params: depth={depth}, n_fs={n_fs}, k_f={k_f}, n_xs={n_xs}, n_zs={n_zs}")

    # zs = gen_zs(n_zs)
    # print(len(zn), len(zn[0]), random.sample(zn, 10))
    # print(zs.shape, zs[400:410])

    # zs, exprs = make_zs_and_exprs()
    zs, exprs = load_zs_and_exprs()
    xs, ys = make_and_store_exs(zs, exprs, n_fs, k_f, n_xs)
    # # xs, ys = load_exs()

    # Format examples
    n_exs = len(xs)
    xs = T.stack([T.reshape(x, (n_xs * 2, 1, B_W, B_H)) for x in xs])
    ys = T.reshape(T.tensor(ys), (n_exs, 1)) # reshape ys into a column vector
    xs = xs.to(device)
    ys = ys.to(device)
    print(xs.shape, ys.shape)

    # Train and test NN
    net = Net(n=n_xs)
    train_nn(net, n_xs, xs, ys, epochs)
    test_nn(net, xs, ys, threshold)

    
