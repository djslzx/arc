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

PATH='./model.pt'

# Tensorboard
import torch.utils.tensorboard as tb

class Net(nn.Module):

    def __init__(self, n, c=6, batch_size=4):
        """
        n: number of elts in each example set
        """
        super(Net, self).__init__()
        self.n = n
        self.c = c
        self.batch_size = batch_size
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
        x: a tensor w/ shape (2N x 1 x Bh x Bw)
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
        return x

    def load(self):
        self.load_state_dict(torch.load(PATH))

    def train(self, data, labels, epochs):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = T.optim.Adam(self.parameters(), lr=0.001)

        dataset = T.utils.data.TensorDataset(data, labels)
        dataloader = T.utils.data.DataLoader(dataset, shuffle=True)

        self.to(device)
        writer = tb.SummaryWriter()
        
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

        T.save(self.state_dict(), PATH)
        print('Finished training')

def make_data(exprs, zs, n_fs, n_reps, subset_size):
    """
    Generate examples [{f(z) | z in a subset of zs} | f is an expr]

    n_fs: number of unique f's to use
    n_reps: number of times to repeat each f
    subset_size: number of xs to generate for each example set
    """
    assert subset_size * n_reps <= len(zs), "not enough zs to sample xs from"

    def score(dist):
        return 1/(1 + dist)

    def eval(f, z):
        try:
            return f.eval({'z': z})
        except AssertionError:
            return T.zeros(B_W, B_H)

    xs = dict()
    for i, f in enumerate(exprs[:n_fs]):
        print(f'Evaluating exprs... [{i+1}/{n_fs}]', end='\r')
        xs[f] = [eval(f, z) for z in zs]
    print()

    data, labels = [], []
    for i, (f, f_xs) in enumerate(xs.items()):
        print(f'Generating positive and negative examples... [{i+1}/{len(xs)}]', end='\r')
        # positive examples
        for p1, p2 in util.chunk_pairs(f_xs, subset_size, n_reps):
            data.append(T.cat((T.stack(p1), T.stack(p2))))
            labels.append(score(0))
        # negative examples
        for i in range(n_reps):
            g = random.choice(list(xs.keys() - {f}))
            g_xs = xs[g]        # modify to use the same zs for different f and g?
            f_subset = T.stack(random.sample(f_xs, subset_size))
            g_subset = T.stack(random.sample(g_xs, subset_size))
            data.append(T.cat((f_subset, g_subset)))
            labels.append(score(f.dist_to(g)))
    print()
    return T.stack(data), T.tensor(labels)

def load_zs_and_exprs():
    print('Loading zs, exprs...')
    exprs = []
    with open('../data/exprs.dat', 'rb') as f: 
        while True:
            try:
                exprs.append(pickle.load(f))
            except EOFError: 
                break
    with open('../data/zs.dat', 'rb') as f: 
        zs = pickle.load(f)
    return zs, exprs
    
def make_and_store_data(exprs, zs, n_fs, n_reps, subset_size):
    print('Making data...')
    xs, ys = make_data(exprs, zs, n_fs, n_reps, subset_size)
    print('Saving examples...')
    with open('../data/exs.dat', 'wb') as f:
        pickle.dump((xs, ys), f)
    return xs, ys

def load_data():
    print('Loading data from file...')
    xs, ys = None, None
    with open('../data/exs.dat', 'rb') as f:
        xs, ys = pickle.load(f)
    return xs, ys

def train_nn(net, xs, ys, epochs):
    print('Training NN...')
    start_time = time.time()
    net.train(xs, ys, epochs)
    end_time = time.time()
    print(f'Took {(end_time - start_time):.5f}s to train.')

def test_nn(net, inputs, outputs, exprs):
    print('Testing NN...')
    n_exs = len(inputs)
    n_correct = 0
    fps, fns, tps, tns = 0, 0, 0, 0

    predictions = []
    for i, o in zip(inputs, outputs):
        label = o.item()
        predicted = net(i).item()
        predictions.append(predicted)

        l = label > 0.5
        p = predicted > 0.5

        fps += not l and p
        fns += l and not p

        n_correct += l == p

    predictions = T.tensor(predictions).to(device)
    print('predictions', tensor_stats(predictions))
    print('scaled predictions', tensor_stats(1/predictions - 1))
    diffs = T.abs(predictions - outputs).to(device)
    print('diffs', tensor_stats(diffs))
    print('labels', tensor_stats(outputs))

    print(f'|correct|: {n_correct}, |exs|: {n_exs}, correct %: {n_correct/n_exs * 100}%')
    print(f'fps: {fps}, fns: {fns}')

    # fp_exs = random.sample(fp_pairs.items(), min(2, fps))
    # fn_exs = random.sample(fn_pairs.items(), min(2, fns))
    # for fp_ex in fp_exs:
    #     print("fp:", fp_ex)
    #     # print(exprs[i//n_reps].pretty_print())
    # for fn_ex in fn_exs:
    #     print("fn:", fn_ex)
    #     # print(exprs[i//n_reps].pretty_print())

def tensor_stats(tensor):
    return f'''stats: 
    max={tensor.max().item()}, 
    min={tensor.min().item()}, 
    mean={tensor.mean().item()}, 
    median={tensor.median().item()}, 
    |{{x = 0}}|={(tensor == 0).sum()},
    |{{x > 0}}|={(tensor > 0).sum()},
    shape={tensor.shape},
    '''

if __name__ == '__main__':
    zs, exprs = load_zs_and_exprs()
    n_fs = 100
    n_reps=  10
    epochs = 1000
    subset_size = 5             # len(zs)/n_reps
    print(f"Params: n_fs={n_fs}, n_reps={n_reps}, subset_size={subset_size}, epochs={epochs}")

    def reshape_data(data):
        return T.stack([T.reshape(x, (subset_size * 2, 1, B_W, B_H)) for x in data]).to(device)
    
    def reshape_labels(labels):
        return T.reshape(labels, (len(labels), 1)).to(device)

    # data, labels = make_and_store_data(exprs, zs, n_fs, n_reps, subset_size)
    data, labels = load_data()

    # Print stats on data, labels
    print(f'data:\n    shape={data.shape}')
    print('labels', tensor_stats(labels))

    # Format examples
    d = len(data)//2
    # labels = 1/(labels + 1)
    train_data, test_data = data[:d], data[d:]
    train_labels, test_labels = labels[:d], labels[d:]

    n_exs = len(train_data)
    train_data = reshape_data(train_data)
    train_labels = reshape_labels(train_labels)
    test_data = reshape_data(test_data)
    test_labels = reshape_labels(test_labels)

    print("train:", train_data.shape, train_labels.shape)
    print("test:", test_data.shape, test_labels.shape)

    # Train and test
    net = Net(n=subset_size).to(device)
    train_nn(net, train_data, train_labels, epochs)
    # net.load_state_dict(T.load('model.pt'))
    test_nn(net, test_data, test_labels, exprs)

    

