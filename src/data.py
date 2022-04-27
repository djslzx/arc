"""
Generate data to train value & policy nets
"""
import pickle
from typing import Optional, List, Tuple, Iterable, Generator, Dict
import multiprocessing as mp

from grammar import *
import util
import viz

Envs = List[Dict]

def render(p, envs):
    return T.stack([p.eval(env) for env in envs])

def to_delta_examples(f: Expr, envs: List[Dict], split_envs=False) \
        -> Generator[Tuple[Tuple[List, Envs, T.Tensor],
                           Tuple[List, Envs, T.Tensor],
                           List],
                     None, None]:
    """
    Converts a closure (f: program, z: environment) into examples of teacher-forcing deltas (B, f', B' -> d')
    for each program f, where:
     - f' is a prefix of f, B' is f'(z), and
     - d' is the next line in f after the lines in f'.
    """
    # TODO: make simplification of indices work with z inference
    f_toks = f.serialize()
    f_lines = f.lines()
    
    # use different sets of envs for prefixes vs full program
    full_env_choices, prefix_env_choices = [envs], [envs]
    if split_envs:
        assert len(envs) % 2 == 0
        n_envs = len(envs) // 2
        full_env_choices = [envs[:n_envs], envs[n_envs:]]
        prefix_env_choices = [envs[:n_envs], envs[n_envs:]]
    
    for f_envs, p_envs in it.product(full_env_choices, prefix_env_choices):
        f_bmps = T.stack([f.eval(env) for env in f_envs])

        for i in range(len(f_lines) + 1):
            p = Seq(*f_lines[:i])
            p_bmps = T.stack([p.eval(env) for env in p_envs])  # well-defined on envs b/c f is
            p_toks = p.serialize()
            d_toks = f_lines[i].serialize() if i < len(f_lines) else [SEQ_END]
            yield (p_toks, p_envs, p_bmps), (f_toks, f_envs, f_bmps), d_toks

def gen_closures(n_envs: int, n_programs: int, n_lines: int,
                 arg_exprs: List[Expr],
                 rand_arg_bounds: Tuple[int, int],
                 line_types: List[type], line_type_weights: List[float],
                 verbose=False):
    """
    Generate `(f, z)` pairs with associated training outputs (value, policy).
    """
    for i in range(n_programs):
        envs = seed_libs(n_envs)
        p = create_program(envs=envs, arg_exprs=arg_exprs, n_lines=n_lines,
                           rand_arg_bounds=rand_arg_bounds,
                           line_types=line_types, line_type_weights=line_type_weights)
        if verbose: print(f'[{i}/{n_programs}]: {p}, {extract_zs(envs)}')
        yield p, envs

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

def create_program(envs: List[dict], arg_exprs: List[Expr], n_lines: int, rand_arg_bounds: Tuple[int, int],
                   line_types: List[type], line_type_weights: List[float] = None,
                   verbose=False):
    """
    Creates a new program matching the spec. Uses expressions containing randomness for arguments
    of each line object, but doesn't allow random colors.
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
        return random.randint(lo, hi)  # TODO: allow non-uniform sampling using random.choices
    
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

def rm_dead_code(lines: List[Expr], envs):
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

def canonical_ordering(lines: List[Expr]):
    
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

def demo_create_program():
    envs = seed_libs(5)
    p = create_program(envs=envs,
                       arg_exprs=[Z(i) for i in range(LIB_SIZE)] + [Num(i) for i in range(0, 10)],
                       n_lines=3,
                       line_types=[Rect, Line, Point],
                       line_type_weights=[4, 3, 1],
                       rand_arg_bounds=(0, 2),
                       verbose=True)
    print(p, extract_zs(envs))
    ps = T.stack([p.eval(env) for env in envs])
    viz.viz_mult(ps, text=p.simplify_indices())

def demo_gen_closures():
    fname = '../data/policy-test-closures/'
    arg_exprs = [Z(i) for i in range(LIB_SIZE)] + [Num(i) for i in range(0, 10)]
    n_envs = 6
    n_workers = 10
    n_programs_per_worker = 10
    workload_sz = n_programs_per_worker * n_workers
    n_lines = 5
    with mp.Pool(processes=n_workers) as pool:
        pool.starmap(gen_closures,
                     [(f'{fname}{i}.dat', n_envs, n_programs_per_worker, n_lines,
                       arg_exprs, (0, 2), [Rect, Line, Point], [4, 3, 1], True)
                      for i in range(workload_sz)])

    for i, (f, envs) in enumerate(util.load_incremental(f'{fname}*.dat')):
        print(i, f)

def gen_closures_and_deltas(worker_id: int, closures_loc: str, deltas_loc: str,
                            n_envs: int, n_programs: int, n_lines: int,
                            arg_exprs: List[Expr],
                            rand_arg_bounds: Tuple[int, int],
                            line_types: List[type], line_type_weights: List[float],
                            split_envs=False,
                            verbose=False):
    util.make_parent_dir(closures_loc)
    util.make_parent_dir(deltas_loc)
    with open(closures_loc, 'wb') as closures_file, open(deltas_loc, 'wb') as deltas_file:
        gen = gen_closures(n_envs=n_envs, n_programs=n_programs, n_lines=n_lines, arg_exprs=arg_exprs,
                           rand_arg_bounds=rand_arg_bounds,
                           line_types=line_types, line_type_weights=line_type_weights)
        for i, (f, envs) in enumerate(gen):
            pickle.dump((f, envs), closures_file)
            for delta in to_delta_examples(f, envs, split_envs=split_envs):
                (p_toks, p_envs, p_bmps), (f_toks, f_envs, f_bmps), d = delta
                print("delta example:", p_toks, p_envs, p_bmps, f_toks, f_envs, f_bmps, d,
                      sep='\n', end='\n\n')
                pickle.dump(delta, deltas_file)
            if verbose: print(f'[{worker_id}][{i}/{n_programs}]: {f}')

def gen_closures_and_deltas_mp(closures_loc_prefix: str, deltas_loc_prefix: str,
                               n_envs: int, n_programs: int, n_lines_bounds: Tuple[int, int],
                               rand_arg_bounds: Tuple[int, int],
                               line_types: List[type], line_type_weights: Optional[List[float]] = None,
                               split_envs=False,
                               n_workers: int = 1):
    """
    Generate a stream of closures and directly convert them to deltas. Write both to files.
    """
    assert n_programs % n_workers == 0, f'Expected n_workers to divide n_programs, got {n_programs}/{n_workers}'
    n_lines_lo, n_lines_hi = n_lines_bounds
    assert n_lines_lo >= 0, f'Expected n_lines_bounds > 1, found {n_lines_bounds}'
    
    # run n_workers workers to generate a total of n_programs programs of each size in n_lines
    arg_exprs = [Z(i) for i in range(LIB_SIZE)] + [Num(i) for i in range(0, 10)]
    n_programs_per_worker = n_programs // n_workers
    with mp.Pool(processes=n_workers) as pool:
        pool.starmap(gen_closures_and_deltas,
                     [(i,
                       f'{closures_loc_prefix}closures_{i}.dat',
                       f'{deltas_loc_prefix}deltas_{i}.dat',
                       n_envs * (2 if split_envs else 1), n_programs_per_worker, n_lines,
                       arg_exprs, rand_arg_bounds, line_types, line_type_weights,
                       split_envs, False)
                      for n_lines in range(n_lines_lo, n_lines_hi + 1)
                      for i in range(n_workers)])
    # separate closure and delta gen? might allow better allocation of workers

# Examine datasets
def collect_stats(dataset: Iterable, max_line_count=3):
    by_len = {
        i: {"count": 0,    # number of programs of length i
            "overlap": 0}  # overlap between rendered lines
        for i in range(1, max_line_count+1)
    }
    n_of_type = {t: 0 for t in [Point, Line, Rect]}  # track number of lines by type
    seen_lines = set()  # track unique lines

    for (p, z_p, b_p), (f, z_f, b_f), d in dataset:
        f = deserialize(f)
        lines = f.lines()
        n_lines = len(lines)
        
        z_f, z_p = z_f[0], z_p[0]
        print(f, z_f, z_p)
        # print(f, lines, n_lines, by_len, n_of_type, seen_lines)
        
        seen_lines.union(lines)
        by_len[n_lines]["count"] += 1

        for line in lines:
            t = type(line)
            n_of_type[t] += 1

        overlap_p = T.sum(T.sum(T.stack([line.eval(z_p) > 0 for line in lines]), dim=-1) > 1).item()
        overlap_f = T.sum(T.sum(T.stack([line.eval(z_f) > 0 for line in lines]), dim=-1) > 1).item()

        by_len[n_lines]["overlap"] = (overlap_p + overlap_f)/(2 * B_H * B_W)
    
    print(f"Unique lines: {len(seen_lines)}",
          f"Number of lines by type: {n_of_type}",
          f"Counts by length:",
          *[f"  {by_len[i]['count']}" for i in range(max_line_count)],
          f"Overlaps by length:",
          *[f"  {by_len[i]['overlap']}" for i in range(max_line_count)],
          sep="\n")


if __name__ == '__main__':
    # demo_gen_program()
    # demo_gen_closures()
    # demo_gen_policy_data()
    
    dir = "../data/policy-pretraining"  # '/home/djl328/arc/data/policy-pretraining'
    code = '1-RLP-5e1l0~1z'
    # t = util.timecode()
    # for mode in ['training',
    #              # 'validation'
    #              ]:
    #     print(f"Generating policy data for mode={mode}")
    #     gen_closures_and_deltas_mp(
    #         closures_loc_prefix=f'{dir}/{code}/{mode}_{t}/',
    #         deltas_loc_prefix=f'{dir}/{code}/{mode}_{t}/',
    #         n_envs=3,
    #         n_programs=1,
    #         n_lines_bounds=(1, 1),
    #         rand_arg_bounds=(1, 1),
    #         line_types=[Rect, Line, Point],
    #         line_type_weights=[4, 3, 1],
    #         n_workers=1,
    #     )
    #     util.join_glob(f"{dir}/{code}/{mode}_{t}/deltas_*.dat",
    #                    f"{dir}/{code}/{mode}_{t}/joined_deltas.dat")

    collect_stats(util.load_incremental("../data/policy-pretraining/100k-RLP-5e1l0~1z/training_deltas.dat"),
                  max_line_count=1)
    collect_stats(util.load_incremental("../data/policy-pretraining/100k-RLP-5e1l0~1z/validation_deltas.dat"),
                  max_line_count=1)