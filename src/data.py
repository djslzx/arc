"""
Generate data to train value & policy nets
"""
import math
import time
import random
import torch as T
import itertools as it
import typing as typ
import multiprocessing as mp

from grammar import *
from bottom_up import bottom_up_generator
import util
import viz

def gen_data(n_envs: int, n_programs: int, n_lines: typ.Union[int, tuple[int, int]],
             rand_arg_bounds: tuple[int, int],
             line_types: list[type], line_type_weights: list[float] = None,
             n_workers: int = 1):
    """
    Samples a set of programs of the form `Seq(l_1, l_2. ..., l_k)`.
    """
    uid = time.time()
    # generate a set of `n_envs` environments
    pass
    
def gen_data_worker(fname: str, n_envs: int, n_programs: int, n_lines: int,
                    arg_exprs: list[Expr],
                    rand_arg_bounds: tuple[int, int],
                    line_types: list[type], line_type_weights: list[float]):
    """
    Generate `(f, z)` pairs with associated training outputs (value, policy).
    """
    util.touch(fname)
    for i in range(n_programs):
        envs = seed_libs(n_envs)

        # choose n_lines lines compatible with env
        lines = gen_lines(envs=envs, arg_exprs=arg_exprs, n_lines=n_lines,
                          rand_arg_bounds=rand_arg_bounds,
                          line_types=line_types, line_type_weights=line_type_weights)
        f = Seq(*lines)
        util.save(data=(f, envs), fname=fname, append=True)

def gen_lines(envs: list[dict], arg_exprs: list[Expr], n_lines: int, rand_arg_bounds: tuple[int, int],
              line_types: list[type], line_type_weights: list[float] = None,
              verbose=False):
    """
    Generates a sequence of lines.

    Uses expressions containing randomness for arguments of each line object, but doesn't allow
    random colors.
    """
    def is_valid(expr):
        for env in envs:
            try:
                expr.eval(env)
            except AssertionError:
                return False
        return True

    def choose_n_rand_args(n_args):
        lo, hi = rand_arg_bounds
        lo = util.clamp(lo, 0, n_args)
        hi = util.clamp(hi, 0, n_args)
        return random.randint(lo, hi)

    assert rand_arg_bounds[0] <= rand_arg_bounds[1], 'Invalid rand argument bounds'
    rand_exprs, const_exprs = util.split(arg_exprs, lambda expr: expr.zs())
    lines = []
    while len(lines) < n_lines:
        n_tries = 0
        line_type = random.choices(population=line_types, weights=line_type_weights)[0]
        n_args = len(line_type.in_types) - 1
        n_rand_args = choose_n_rand_args(n_args)
        n_const_args = n_args - n_rand_args

        # try choosing valid args for the line type
        # note: assumes that all in_types are integers (not true in general, but true of shapes)
        line = None
        while line is None:
            n_tries += 1
            if verbose and n_tries % 1000 == 0: print(f'[{len(lines)}, {line_type.__name__}]: {n_tries} tries')
            
            const_args = random.choices(population=const_exprs, k=n_const_args)
            rand_args = random.choices(population=rand_exprs, k=n_rand_args)
            args = util.shuffled(const_args + rand_args)
            cand = line_type(*args, color=Num(random.randint(0, 9)))
            if is_valid(cand):
                line = cand
        lines.append(line)
        lines = canonical_ordering(lines)
        lines = rm_dead_code(lines, envs)
        if verbose: print(f'[{len(lines)}, {line_type.__name__}]: {n_tries} tries')
    return lines

def rm_dead_code(lines, envs):

    def render(xs):
        return T.stack([Seq(*xs).eval(env) for env in envs])

    orig_bmps = render(lines)
    out = []
    for i, line in enumerate(lines):
        excl_bmps = render(out + lines[i + 1:])
        if not T.equal(excl_bmps, orig_bmps):
            out.append(line)
    return out

def canonical_ordering(lines):

    def destructure(e):
        if isinstance(e, Point):
            return 0, e.x, e.y
        elif isinstance(e, Line):
            return 1, e.x1, e.y1, e.x2, e.y2
        elif isinstance(e, Rect):
            return 2, e.x_min, e.y_min, e.x_max, e.y_max
        
    return sorted(lines, key=destructure)

def extract_zs(envs):
    return [env['z'] for env in envs]

def demo_gen_lines():
    environments = seed_libs(5)
    ls = gen_lines(envs=environments,
                   arg_exprs=[Z(i) for i in range(LIB_SIZE)] + [Num(i) for i in range(0, 10)],
                   n_lines=3,
                   line_types=[Rect, Line, Point],
                   line_type_weights=[4, 3, 1],
                   rand_arg_bounds=(0, 2),
                   verbose=True)
    print(ls, extract_zs(environments))
    p = Seq(*ls)
    ps = T.stack([
        p.eval(env) for env in environments
    ])
    viz.viz_mult(ps, text=p.simplify_indices())

def demo_gen_data_worker():
    fname = '../data/test1.dat'
    gen_data_worker(fname=fname,
                    n_envs=5, n_programs=10, n_lines=3,
                    arg_exprs=[Z(i) for i in range(LIB_SIZE)] + [Num(i) for i in range(0, 10)],
                    rand_arg_bounds=(0, 2),
                    line_types=[Rect, Line, Point],
                    line_type_weights=[4, 3, 1])
    for f, envs in util.load_incremental(fname):
        print(f, extract_zs(envs))


if __name__ == '__main__':
    # demo_gen_lines()
    demo_gen_data_worker()
