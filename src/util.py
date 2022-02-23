import torch as T
import pickle
import os
from random import choice, shuffle

dirname = os.path.dirname(__file__)

def shuffled(s):
    shuffle(s)
    return s

def to_toks(s):
    def tonum(s):
        try: return int(s)
        except: return s

    return [tonum(c) for c in s]

def clamp(x, lo, hi):
    assert hi >= lo
    if x > hi:   return hi
    elif x < lo: return lo
    return x

def avg(it):
    s = 0
    n = 0
    for x in it:
        s += x
        n += 1
    return s/n

def sum_sq(it):
    return sum(x*x for x in it)

def chunks(l, k, n):
    size = len(l)
    for i in range(n):
        start = (i * k) % size
        end = (start + k) % size
        yield l[start:end] if start < end else l[start:] + l[:end]

def chunk_pairs(l, k, n):
    """
    Iterator over n k-elt chunks of list l, yielding pairs of adjacent chunks
    """
    size = len(l)
    for i in range(n):
        start = (i * k) % size
        mid = (start + k) % size
        end = (mid + k) % size
        
        yield (l[start:mid] if start < mid else l[start:] + l[:mid],
               l[mid:end] if mid < end else l[mid:] + l[:end])

def img_to_tensor(lines, w=-1, h=-1):
    """Converts a list of strings into a float tensor"""
    if not lines: return T.Tensor([])

    lines_l = len(lines)
    lines_w = max(len(line) for line in lines)

    if h == -1: h = lines_l
    if w == -1: w = lines_w

    def cell(x, y):
        if y < lines_l and x < lines_w:
            try: 
                return int(lines[y][x])
            except ValueError:
                return int(lines[y][x] == '#')
            except IndexError:
                return 0
        else:
            return 0

    return T.tensor([[cell(x, y) for x in range(w)] 
                     for y in range(h)]).float()

def tensor_to_pts(tensor):
    return [(x,y) for x,y in tensor.nonzero().tolist()]

def unzip(l):
    return tuple(list(x) for x in zip(*l))


def make_bitmap(f, W, H):
    return T.tensor([[f((x, y))
                      for x in range(W)]
                     for y in range(H)]).float()

def split(l, pred):
    """
    Split l into two lists `a` and `b`, where
    all elts of `a` satisfy `pred` and all elts of `b` do not
    """
    sat, unsat = [], []
    for x in l:
        if pred(x): sat.append(x)
        else:       unsat.append(x)
    return sat, unsat

def to_abspath(path):
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(dirname, path)

def save(data, fname, append=False, verbose=True):
    path = to_abspath(fname)
    if verbose: print(f'Saving to {path}...')
    mode = 'ab' if append else 'wb'
    with open(path, mode) as f:
        pickle.dump(data, f)

def load(fname, verbose=True):
    path = to_abspath(fname)
    if verbose: print(f'Loading from {path}...')
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_incremental(fname, verbose=True):
    path = to_abspath(fname)
    if verbose: print(f'Loading from {path}...')
    with open(path, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

if __name__ == '__main__':
    print(img_to_tensor(['_#_',
                         '#_#',
                         '__#']))
    print(img_to_tensor(['_#_',
                         '#_#',
                         '__#'], 2, 2))
    print(img_to_tensor(['_#_',
                         '#_#',
                         '__#'], 4, 4))
    print(img_to_tensor(['_1_',
                         '2_2',
                         '__3'], 4, 4))
