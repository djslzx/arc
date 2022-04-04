"""
Generate data to train value & policy nets
"""
import pickle
from datetime import datetime
from typing import Optional, List, Tuple
import multiprocessing as mp

from grammar import *
import util
import viz


def render(p, envs):
    return T.stack([p.eval(env) for env in envs])

def policy_data_to_examples(data_src: str, split_envs: bool = False):
    """
    Converts a list of (f: program, z: environment) tuples into examples of
    teacher-forcing deltas (B, f', B' -> d') for each program f,
    where f' is a prefix of f, B' is f'(z), and d' is the next line in f after the lines in f'.
    """
    for (f, envs) in util.load_incremental(data_src):
        f_simplified_lines = f.simplify_indices().lines()  # f w/ simplified indices
        f_lines = f.lines()
        
        # use different sets of envs for prefixes vs full program
        full_env_choices, prefix_env_choices = [envs], [envs]
        if split_envs:
            assert len(envs) % 2 == 0
            n_envs = len(envs) // 2
            full_env_choices = [envs[:n_envs], envs[n_envs:]]
            prefix_env_choices = [envs[:n_envs], envs[n_envs:]]

        for full_envs, prefix_envs in it.product(full_env_choices, prefix_env_choices):
            # generate full program's bitmaps
            bitmaps = T.stack([f.eval(env) for env in full_envs])
    
            # empty prefix
            empty_bitmaps = T.stack([Seq().eval(env) for env in full_envs])
            yield bitmaps, empty_bitmaps, [], Seq().serialize()
        
            # nonempty prefixes
            for i in range(len(f_lines)):
                prefix = Seq(*f_lines[:i])
                prefix_bitmaps = T.stack([prefix.eval(env) for env in prefix_envs])  # well-defined on envs b/c f is
                prefix_tokens = Seq(*f_simplified_lines[:i]).serialize()
                delta = f_simplified_lines[i].serialize()
                yield bitmaps, prefix_bitmaps, prefix_tokens, delta
            
def save_policy_dat_as_examples(data_src: str, save_loc: str, split_envs: bool = True, verbose: bool = False):
    util.make_parent_dir(save_loc)
    with open(save_loc, 'wb') as file:
        for item in policy_data_to_examples(data_src, split_envs=split_envs):
            if verbose: print(item)
            pickle.dump(item, file)

def gen_policy_data(fname_prefix: str,
                    n_envs: int, n_programs: int, n_lines_bounds: Tuple[int, int],
                    rand_arg_bounds: Tuple[int, int],
                    line_types: List[type], line_type_weights: Optional[List[float]] = None,
                    n_workers: int = 1):
    """
    Generates data for pretraining policy net by sampling a set of programs of the form
    `Seq(l_1, l_2. ..., l_k)`, where k is in n_lines_bounds
    """
    assert n_programs % n_workers == 0, f'Expected n_workers to divide n_programs, got {n_programs}/{n_workers}'
    n_lines_lo, n_lines_hi = n_lines_bounds
    assert n_lines_lo >= 0, f'Expected n_lines_bounds > 1, found {n_lines_bounds}'
    
    # run n_workers workers to generate a total of n_programs programs of each size in n_lines
    arg_exprs = [Z(i) for i in range(LIB_SIZE)] + [Num(i) for i in range(0, 10)]
    n_programs_per_worker = n_programs // n_workers
    with mp.Pool(processes=n_workers) as pool:
        pool.starmap(worker_gen_policy_data,
                     [(f'{fname_prefix}/{i}.dat', n_envs * 2, n_programs_per_worker, n_lines,
                       arg_exprs, rand_arg_bounds, line_types, line_type_weights, True)
                      for n_lines in range(n_lines_lo, n_lines_hi+1)
                      for i in range(n_workers)])
    
def worker_gen_policy_data(fname: str, n_envs: int, n_programs: int, n_lines: int,
                           arg_exprs: List[Expr],
                           rand_arg_bounds: Tuple[int, int],
                           line_types: List[type], line_type_weights: List[float],
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
            if verbose: print(f'[{i}/{n_programs}]: {p}, {extract_zs(envs)}')
            pickle.dump((p, envs), file)

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

def gen_program(envs: List[dict], arg_exprs: List[Expr], n_lines: int, rand_arg_bounds: Tuple[int, int],
                line_types: List[type], line_type_weights: List[float] = None,
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
    """
    Remove any lines that don't affect the render of Seq(lines) in any of the provided environments.
    
    Exploits the fact that earlier lines in the sequence are rendered first.
    """
    orig_bmps = render(Seq(*lines), envs)
    out = []
    for i, line in enumerate(lines):
        excl_lines = out + lines[i + 1:]
        excl_bmps = render(Seq(*excl_lines), envs)
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

    for i, (f, envs) in enumerate(util.load_incremental(f'{fname}*.dat')):
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
    
    dir = '/home/djl328/arc/data/policy-pretraining'
    code = '1mil-RLP-5e1~4l0~2z'
    t = datetime.now().strftime("%b%d_%y_%H-%M-%S")
    for mode in ['training', 'validation']:
        print(f"Generating policy data for mode={mode}")
        gen_policy_data(fname_prefix=f'{dir}/{code}/{mode}_{t}',
                        n_envs=5,
                        n_programs=1_000_000,
                        n_lines_bounds=(1, 3),
                        rand_arg_bounds=(0, 2),
                        line_types=[Rect, Line, Point],
                        line_type_weights=[4, 3, 1],
                        n_workers=100)

    for mode in ['training', 'validation']:
        print(f"Generating examples from policy data for mode={mode}")
        save_policy_dat_as_examples(data_src=f'{dir}/{code}/{mode}_{t}/*.dat',
                                    save_loc=f'{dir}/{code}/{mode}_{t}.exs',
                                    split_envs=True,
                                    verbose=True)
