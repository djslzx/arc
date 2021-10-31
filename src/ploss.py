"""
Perceptual loss

- [x] Generate random expressions in DSL
  - pick a grammar element at random, then recursively pick subcomponents if nonterminal
- Generate random choices of Z
- Use Pytorch to train NN to measure distance in f-space:
  - convolve each example x_i to get a vector
  - use max pooling to combine multiple x_i's into a vector
  - train NN to learn that two vectors are similar if the f's that generated them are similar
    - extension: use some notion of distance between expressions: edit distance?
"""

import random
import math
import util
from grammar import *
from bottom_up import *
from learn import *

def rec_type(t):
    return t.return_type in t.argument_types

def is_const(grammar_elt):
    """Check whether `grammar_elt` (an op or const) is a const"""
    return not grammar_elt.argument_types or type(grammar_elt) in [Zb, Zn]

# def gen_expr_w_root(grammar, root, max_depth):

def gen_expr(grammar, return_type, max_depth):
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
                              (max_depth > 1 or not rec_type(e))])
        if is_const(root):
            return root
        else:
            args = [gen_expr(grammar, arg_type, max_depth - 1)
                    for arg_type in root.argument_types]
            return root(*args)

def train_nn():
    pass

if __name__ == '__main__':
    g = Grammar(ops=[Plus, Minus, Times, Rect, Program],
                consts=[Zn(Num(i)) for i in range(Z_SIZE)] + 
                       [Num(i) for i in range(Z_LO, Z_HI + 1)])
    for i in range(1000):
        expr = gen_expr(g, "Bitmap", 10)
        print(expr.pretty_print())
        # val = expr.eval({})

