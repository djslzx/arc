import torch as T
import torch.nn.functional as F
import pickle
import pdb
import os
from glob import glob
from math import floor, ceil
from pathlib import Path
import random
from datetime import datetime
from typing import List

dirname = os.path.dirname(__file__)

def pad_tensor(t: T.Tensor, h: int, w: int, padding_token: int = 0) -> T.Tensor:
    """Pad t to height h and width w"""
    assert (dims := len(t.shape)) == 2, \
        f'Expected a 2-dimensional tensor, but got a {dims}-dimensional tensor'
    assert h >= t.size(0) and w >= t.size(1), \
        f'Padded dimensions are smaller than tensor size: h={h}, w={w}, t.shape={t.shape}'
    return F.pad(t, (0, w - t.size(1), 0, h - t.size(0)), value=padding_token)

def fill_height(t: T.Tensor) -> int:
    assert len(t.shape) >= 2
    return (t.sum(dim=1) > 0).sum().item()

def fill_width(t: T.Tensor) -> int:
    assert len(t.shape) >= 2
    return (t.sum(dim=0) > 0).sum().item()

def uniq(l: List) -> List:
    joined = []
    for x in l:
        if x not in joined:
            joined.append(x)
    return joined

def add_channels(img, n_classes=10):
    """
    Turns a W x H tensor into a C x W x H tensor, where the C dimension is a one-hot encoding for the color
    values in the image
    """
    return F.one_hot(img, n_classes).transpose(-1, -2).transpose(-2, -3)

def unwrap_tensor(t):
    if isinstance(t, T.Tensor):
        return t.item()
    else:
        return t

def wrap_tensor(t):
    if isinstance(t, T.Tensor):
        return t
    else:
        return T.Tensor(t)

def timecode() -> str:
    """Returns the timecode for the current time"""
    return datetime.now().strftime("%b%d_%y_%H-%M-%S")

def pad(v, length: int, value: int):
    return F.pad(v, pad=(0, length - len(v)), value=value)

def filter_top_p(v, p=0.95):
    x = v.clone()
    values, indices = T.sort(x, descending=True)
    sums = T.cumsum(values, dim=-1)
    mask = sums >= p
    # right-shift indices to keep first sum >= p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    # filter out elements in v
    for b in range(x.shape[0]):
        x[b, indices[b, mask[b]]] = 0
    return x

def shuffled(it):
    random.shuffle(it)
    return it

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
    if x > hi: return hi
    if x < lo: return lo
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

def make_parent_dir(path):
    """Make parent dir if it doesn't already exist"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def save(data, fname, append=False, verbose=True):
    path = to_abspath(fname)
    if verbose: print(f'Saving to {path}...')
    mode = 'ab' if append else 'wb'
    make_parent_dir(path)
    with open(path, mode) as f:
        pickle.dump(data, f)

def touch(fname):
    path = to_abspath(fname)
    make_parent_dir(path)
    f = open(path, 'wb')
    f.close()

def load(fname, verbose=True):
    abspath = to_abspath(fname)
    assert glob(abspath), f'Found empty glob: {abspath} -> {glob(abspath)}'
    path = glob(abspath)[0]
    if verbose: print(f'Loading from {path}...')
    with open(path, 'rb') as f:
        return pickle.load(f)
        
def load_incremental(file_glob, verbose=False):
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

def join_glob(in_glob: str, out: str):
    """Join all globbed files into a single file"""
    in_fnames = glob(in_glob)
    make_parent_dir(out)
    print(f"Joining {in_fnames} => {out}...")
    with open(out, 'wb') as f_out:
        for fname in in_fnames:
            print(f"Reading from {fname}...")
            with open(fname, 'rb') as f_in:
                while True:
                    try:
                        x = pickle.load(f_in)
                        pickle.dump(x, f_out)
                    except EOFError:
                        break
                        
def weave_glob(glob_str: str, out: str):
    """
    Join all lines in globbed files by picking lines at random from each of the files and repeating.
    """
    filenames = glob(glob_str)
    assert len(filenames) >= 1, f'Found no files for glob={glob_str}'
    make_parent_dir(out)
    print(f"Weaving {len(filenames)} files into {out}...")
    with open(out, 'wb') as f_out:
        file_objs = [open(filename, 'rb') for filename in filenames]
        while file_objs:
            f_in = random.choice(file_objs)
            try:
                x = pickle.load(f_in)
                pickle.dump(x, f_out)
            except EOFError:
                file_objs.remove(f_in)

def print_glob_contents(glob_str: str, verbose=False):
    filenames = glob(glob_str)
    for filename in filenames:
        if verbose: print(f'START {filename}')
        with open(filename, 'rb') as f:
            while True:
                try:
                    print(pickle.load(f))
                except EOFError:
                    break
        if verbose: print(f'END {filename}')

def test_add_channels():
    cases = [
        (T.tensor([[[1]],  # image 1
                   [[2]],  # image 2
                   [[0]]   # image 3
                  ]),
         T.tensor([[[[0]], [[1]], [[0]]],  # image 1
                   [[[0]], [[0]], [[1]]],  # image 2
                   [[[1]], [[0]], [[0]]]   # image 3
                  ])
         ),
        (T.tensor([[[0, 0, 0],
                    [2, 1, 1]]]),
         T.tensor([[[[1, 1, 1],
                     [0, 0, 0]],
                    [[0, 0, 0],
                     [0, 1, 1]],
                    [[0, 0, 0],
                     [1, 0, 0]]]])
         ),
        (T.tensor([
            [  # batch 1
                [[0, 1]],  # image 1
                [[1, 0]]   # image 2
            ],
            [  # batch 2
                [[1, 1]],  # image 1
                [[0, 0]]   # image 2
            ],
            [  # batch 3
                [[1, 1]],  # image 1
                [[0, 0]]  # image 2
            ]
        ]),
         T.tensor([
            [  # batch 1
                [[[1, 0]], [[0, 1]]],  # image 1
                [[[0, 1]], [[1, 0]]]   # image 2
            ],
            [  # batch 2
                [[[0, 0]], [[1, 1]]],  # image 1
                [[[1, 1]], [[0, 0]]]   # image 2
            ],
            [  # batch 3
                [[[0, 0]], [[1, 1]]],  # image 1
                [[[1, 1]], [[0, 0]]]   # image 2
            ]
         ])),
    ]
    for img, ans in cases:
        out = add_channels(img, -1)
        assert T.equal(out, ans), f'Expected {ans} of shape {ans.shape}; ' \
                                  f'Got {out} of shape {out.shape}.'
    print("[+] test_add_channels passed")

def test_fill_measure():
    cases = [
        (img_to_tensor(["###  "]), 1, 3),
        (img_to_tensor(["###  ",
                        "##   "]), 2, 3),
        (img_to_tensor([" ####",
                        "##   "]), 2, 5),
    ]
    for mat, height, width in cases:
        h = fill_height(mat)
        w = fill_width(mat)
        assert height == h and width == w, \
            f"[-] failed test_fill_measure:\n" \
            f"  Expected height={height} and width={width}, but got {h, w}"
    print("[+] passed test_fill_measure")

def test_pad_tensor():
    cases = [
        (T.Tensor([[1, 1],
                   [2, 1]]),
         3, 4, 0,
         T.Tensor([[1, 1, 0, 0],
                   [2, 1, 0, 0],
                   [0, 0, 0, 0]])),
        (T.Tensor([[1, 1, 2],
                   [2, 1, 3]]),
         5, 3, 9,
         T.Tensor([[1, 1, 2],
                   [2, 1, 3],
                   [9, 9, 9],
                   [9, 9, 9],
                   [9, 9, 9]])),
    ]
    for t, h, w, tok, ans in cases:
        out = pad_tensor(t, h, w, tok)
        assert T.equal(out, ans), f"Expected {ans}, but got {out}"
    print("[+] passed test_pad_tensor")

if __name__ == '__main__':
    test_add_channels()
    test_fill_measure()
    test_pad_tensor()
