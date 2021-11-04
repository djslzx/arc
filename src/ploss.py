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

import random
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import util
from grammar import *
from bottom_up import *
from learn import *

def is_recursive(t):
    return t.return_type in t.argument_types

def is_const(grammar_elt):
    """Check whether `grammar_elt` (an op or const) is a const"""
    return not grammar_elt.argument_types or type(grammar_elt) in [Zb, Zn]

def gen_expr_w_root(grammar, root, max_depth, zs=[]):
    if max_depth == 0:
        assert is_const(root), f"Root at depth 0 should be a const, but root={root}"
        return root
    else:
        while True:
            args = [gen_expr(grammar, arg_type, max_depth - 1)
                    for arg_type in root.argument_types]
            out = root(*args)
            if not zs or all(out.satisfies_invariants({"z_n": z}) for z in zs):
                return out

def gen_expr(grammar, return_type, max_depth, zs=[]):
    """
    Generate a random expression from a grammar
    """
    if max_depth == 0:
        consts = [c for c in grammar.consts
                  if c.return_type == return_type]
        return random.choice(consts)
    else:
        root = random.choice([e for e in grammar.ops + grammar.consts 
                              if e.return_type == return_type and
                              (max_depth > 1 or not is_recursive(e))])
        if is_const(root): return root
        else:
            while True:
                args = [gen_expr(grammar, arg_type, max_depth - 1)
                        for arg_type in root.argument_types]
                out = root(*args)
                if not zs or all(out.satisfies_invariants({"z_n": z}) for z in zs):
                    return out

def make_exs(grammar, root, n_fs, mult_fs, n_zs, n_xs, max_depth):
    """
    Generate exs = [{f(z)} | f is an expr]

    max_depth: maximimum depth for f
    n_fs: number of examples to generate
    mult_fs: number of times to repeat each f
    n_zs: number of zs to pick from
    n_xs: number of xs to generate for each example
    """
    # Make a large population of z's, then pick samples (subset) for each f
    # Use subset to 

    # generate zs first so we can evaluate fs wrt zs to ensure soundness
    # TODO: ensure soundness
    zs = [gen_zn() for _ in range(n_zs)]
    fs = [gen_expr_w_root(g, root, max_depth) for _ in range(n_fs)] * mult_fs
    return [(f, [f.eval({"z_n": z}) for z in random.sample(zs, n_xs)]) 
            for f in fs]

class Net(nn.Module):

    def __init__(self, n, c=6, w=B_W, h=B_H):
        """
        n: number of elts in each example set
        """
        super(NeuralNetwork, self).__init__()
        self.n = n
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, c, 3, padding='same')
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
        x = self.conv_stack(x)

        # Split, max pool, flatten, and concat
        a, b = T.split(x, self.n)
        a = T.max(a, dim=0).values
        b = T.max(b, dim=0).values
        x = T.cat((a, b))

        x = T.flatten(x)
        x = self.linear_stack(x)
        return x

    def train(self, xs, ys):
        # TODO

        for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                
        print('Finished Training')

if __name__ == '__main__':
    g = Grammar(ops=[Plus, Minus, Times, Rect, Program],
                consts=[Zn(Num(i)) for i in range(Z_SIZE)] + 
                       [Num(i) for i in range(Z_LO, Z_HI + 1)])
    # for i in range(1000):
    #     expr = gen_expr_w_root(g, Program, 10)
    #     print(expr.pretty_print())
    exs = make_exs(g, Program, n_fs=1, mult_fs=3, n_zs=1000, n_xs=10, max_depth=2)
    for i, (f, xs) in enumerate(exs):
        print(f"f_{i}: {f.pretty_print()}")
        for x in xs:
            print(x.pretty_print())
            print()
