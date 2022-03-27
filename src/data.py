"""
Generate data to train value & policy nets
"""
import pickle
import time
import typing as typ
import multiprocessing as mp

from grammar import *
import util
import viz


def gen_policy_data(fname_prefix: str,
                    n_envs: int, n_programs: int, n_lines_bounds: tuple[int, int],
                    rand_arg_bounds: tuple[int, int],
                    line_types: list[type], line_type_weights: typ.Optional[list[float]] = None,
                    n_programs_per_worker: int = 100):
    """
    Generates data for pretraining policy net by sampling a set of programs of the form
    `Seq(l_1, l_2. ..., l_k)`, where k is in n_lines_bounds
    """
    n_lines_lo, n_lines_hi = n_lines_bounds
    assert n_lines_lo >= 0, f'Expected n_lines_bounds > 1, found {n_lines_bounds}'
    
    fname = f'{fname_prefix}policy-{int(time.time())}'
    arg_exprs = [Z(i) for i in range(LIB_SIZE)] + [Num(i) for i in range(0, 10)]
    
    # run n_workers workers to generate a total of n_programs programs of each size in n_lines
    n_workers = n_programs // n_programs_per_worker + (n_programs % n_programs_per_worker > 0)
    
    with mp.Pool(processes=n_workers) as pool:
        pool.starmap(worker_gen_policy_data,
                     [(f'{fname}/{i}.dat', n_envs, n_programs_per_worker, n_lines,
                       arg_exprs, rand_arg_bounds, line_types, line_type_weights, True)
                      for n_lines in range(n_lines_lo, n_lines_hi+1)
                      for i in range(n_workers)])
    
def worker_gen_policy_data(fname: str, n_envs: int, n_programs: int, n_lines: int,
                           arg_exprs: list[Expr],
                           rand_arg_bounds: tuple[int, int],
                           line_types: list[type], line_type_weights: list[float],
                           verbose: bool = False):
    """
    Generate `(f, z)` pairs with associated training outputs (value, policy).
    """
    util.make_parent_dir(fname)
    with open(fname, 'wb') as file:
        for i in range(n_programs):
            envs = seed_libs(n_envs)
            p = gen_program(envs=envs, arg_exprs=arg_exprs, n_lines=n_lines,
                            rand_arg_bounds=rand_arg_bounds,
                            line_types=line_types, line_type_weights=line_type_weights)
            bitmaps = T.stack([p.eval(env) for env in envs])
            p = p.simplify_indices()
            if verbose: print(f'[{i}/{n_programs}]: {p}')
            pickle.dump((p, envs, bitmaps), file)

def compute_value(expr, bitmaps):
    """
    Compute vector of values outputted by value function.
    - log P(B|f')
    - I[f' is a prefix of f]
    - min{g: f'g ~ f} |g|
      - meaning: minimum number of lines that need to be added to f' to yield the same behavior as f
    - min{ d1..dk st f'd1..dk ~ f} -log (
        P(d1 | B, f', B'(f')) *
        P(d2 | B, f'd1, B'(f'd1)) *
        P(d3 | B, f'd1d2, B'(f'd1d2)) *
        ...)
    """
    pass

def gen_program(envs: list[dict], arg_exprs: list[Expr], n_lines: int, rand_arg_bounds: tuple[int, int],
                line_types: list[type], line_type_weights: list[float] = None,
                verbose=False):
    """
    Generates a program.

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
    
    def choose_n_rand_args(cap):
        lo, hi = rand_arg_bounds
        lo = util.clamp(lo, 0, cap)
        hi = util.clamp(hi, 0, cap)
        return random.randint(lo, hi)
    
    assert rand_arg_bounds[0] <= rand_arg_bounds[1], 'Invalid rand argument bounds'
    rand_exprs, const_exprs = util.split(arg_exprs, lambda expr: expr.zs())
    lines = []
    while len(lines) < n_lines:
        n_tries = 0
        line_type = random.choices(population=line_types, weights=line_type_weights)[0]
        n_args = len(line_type.in_types) - 1
        
        # try choosing valid args for the line type
        # note: assumes that all in_types are integers (not true in general, but true of shapes)
        line = None
        while line is None:
            n_tries += 1
            if verbose and n_tries % 1000 == 0: print(f'[{len(lines)}, {line_type.__name__}]: {n_tries} tries')
            n_rand_args = choose_n_rand_args(n_args)
            n_const_args = n_args - n_rand_args
            rand_args = random.choices(population=rand_exprs, k=n_rand_args)
            const_args = random.choices(population=const_exprs, k=n_const_args)
            args = util.shuffled(const_args + rand_args)
            cand = line_type(*args, color=Num(random.randint(0, 9)))
            if is_valid(cand):
                line = cand
        lines.append(line)
        lines = canonical_ordering(lines)
        lines = rm_dead_code(lines, envs)
        if verbose: print(f'[{len(lines)}, {line_type.__name__}]: {n_tries} tries')
    return Seq(*lines)

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

def demo_gen_program():
    environments = seed_libs(5)
    p = gen_program(envs=environments,
                    arg_exprs=[Z(i) for i in range(LIB_SIZE)] + [Num(i) for i in range(0, 10)],
                    n_lines=3,
                    line_types=[Rect, Line, Point],
                    line_type_weights=[4, 3, 1],
                    rand_arg_bounds=(0, 2),
                    verbose=True)
    print(p, extract_zs(environments))
    ps = T.stack([
        p.eval(env) for env in environments
    ])
    viz.viz_mult(ps, text=p.simplify_indices())

def demo_worker_gen_policy_data():
    fname = '../data/policy-gen-test/'
    arg_exprs = [Z(i) for i in range(LIB_SIZE)] + [Num(i) for i in range(0, 10)]
    n_envs = 6
    n_workers = 10
    n_programs_per_worker = 10
    workload_sz = n_programs_per_worker * n_workers
    n_lines = 5
    with mp.Pool(processes=n_workers) as pool:
        pool.starmap(worker_gen_policy_data,
                     [(f'{fname}{i}.dat', n_envs, n_programs_per_worker, n_lines,
                       arg_exprs, (0, 2), [Rect, Line, Point], [4, 3, 1], True)
                      for i in range(workload_sz)])

    for i, (f, envs, bitmaps) in enumerate(util.load_glob_incremental(f'{fname}*.dat')):
        print(i, f)

def demo_gen_policy_data():
    gen_policy_data(n_envs=5,
                    n_programs=1000,
                    n_lines_bounds=(1, 5),
                    rand_arg_bounds=(0, 2),
                    line_types=[Rect, Line, Point],
                    line_type_weights=[4, 3, 1],
                    fname_prefix='../data/policy-dat-test/')


if __name__ == '__main__':
    # demo_gen_program()
    # demo_worker_gen_policy_data()
    # demo_gen_policy_data()
    
    gen_policy_data(fname_prefix='../data/policy-dat/1mil-RLP-9e1~4l0~2z',
                    n_envs=9,
                    n_programs=1_000_000,
                    n_lines_bounds=(1, 4),
                    rand_arg_bounds=(0, 2),
                    line_types=[Rect, Line, Point],
                    line_type_weights=[4, 3, 1])
    