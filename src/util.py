import torch as T
import pickle
import os
from glob import glob
from math import floor, ceil
from pathlib import Path

dirname = os.path.dirname(__file__)

def is_prefix(l1, l2):
    if len(l1) > len(l2):
        return False
    for x1, x2 in zip(l1, l2):
        if x1 != x2:
            return False
    return True

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

def chunk(n, k):
    """
    Returns a partition of n items into k chunks.

    Output: a list of lengths, where the i-th length is the length of the i-th chunk
    e.g. chunk(10, 3) --> [4, 3, 3]
    """
    return [ceil(n/k) if i < n % k else floor(n/k)
            for i in range(k)]

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
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # make parent dir if it doesn't already exist
    with open(path, mode) as f:
        pickle.dump(data, f)

def load(fname, verbose=True):
    abspath = to_abspath(fname)
    assert glob(abspath), f'Found empty glob: {abspath} -> {glob(abspath)}'
    path = glob(abspath)[0]
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
                
def load_multi_incremental(file_glob, verbose=False):
    """Iterate over multiple files with the prefix"""
    files = glob(file_glob)
    assert files, f'Found empty file glob: {file_glob} -> {files}'
    if verbose:
        print("Globbed files: [")
        for file in files:
            print(f"  {file}")
        print("]")
    
    def gen():
        for file in files:
            with open(file, 'rb') as f:
                while True:
                    try:
                        yield pickle.load(f)
                    except EOFError:
                        break
    return gen()


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
