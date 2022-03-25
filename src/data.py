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
             line_types: list[type],
             n_workers: int = 1):
    """
    Samples a set of programs of the form `Seq(l_1, l_2. ..., l_k)`.
    
    :param n_envs: the number of environments to use in validating each generated program
    :param n_programs: the total number of programs to generate per size
    :param n_lines: the number (or a range) of lines to include in each program
    :param line_types: the kinds of geometric objects to allow (Lines, Rects, Points)
    :param n_workers: the number of worker processes to use (multiprocessing)
    """
    uid = time.time()
    # generate a set of `n_envs` environments
    pass
    
def gen_data_worker(n_envs: int, n_programs: int, n_lines: int, line_types: list[type],
                    cmp_exprs: list[Expr]):
    """
    Generate `(f, z)` pairs with associated training outputs (value, policy).
    """
    for i in range(n_programs):
        envs = seed_libs(n_envs)

        # choose n_lines lines compatible with env
        lines = gen_lines(envs, cmp_exprs, n_lines, line_types)
        f = Seq(*lines)
    pass

def gen_lines(envs: list[dict], cmp_exprs: list[Expr], n_lines: int,
              line_types: list[type], line_type_weights: list[float] = None,
              verbose=False):
    def is_valid(expr):
        for env in envs:
            try:
                expr.eval(env)
            except AssertionError:
                return False
        return True

    lines = []
    while len(lines) < n_lines:
        n_tries = 0
        line_type = random.choices(population=line_types, weights=line_type_weights)[0]
        in_types = line_type.in_types
        # try choosing valid args for the line type
        # note: assumes that all in_types are integers (not true in general, but true of shapes)
        line = None
        while line is None:
            n_tries += 1
            args = random.choices(population=cmp_exprs, k=len(in_types))
            cand = line_type(*args)
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


if __name__ == '__main__':
    environments = seed_libs(5)
    ls = gen_lines(envs=environments,
                   cmp_exprs=[Z(i) for i in range(LIB_SIZE)] + [Num(i) for i in range(0, 10)],
                   n_lines=3,
                   line_types=[Rect, Line, Point],
                   line_type_weights=[5, 2, 1])
    print(ls, [env['z'] for env in environments])
    p = Seq(*ls)
    ps = T.stack([
        p.eval(env) for env in environments
    ])
    viz.viz_mult(ps, text=p)
    