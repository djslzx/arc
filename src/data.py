"""
Generate data to train value & policy nets
"""
import pdb
import pickle
import random
from typing import Optional, List, Tuple, Iterable, Generator, Dict
import multiprocessing as mp
import torch.utils.tensorboard as tb
import math
import sys

from grammar import *
import util
import viz


def render(p, envs):
    return T.stack([p.eval(env) for env in envs])

def filter_zs(env, zs):
    return {'z': T.tensor([z if i in zs else Z_IGNORE
                           for i, z in enumerate(env['z'])])}

def mask_envs(f, envs):
    """
    Mask out z's that aren't used by setting them to Z_IGNORE
    """
    used_z_indices = f.zs()
    n_used = len(used_z_indices)
    # reorder the envs to place used_zs at the front (where they've been remapped) and blank out all other z's
    envs = [{'z': [util.unwrap_tensor(env['z'][idx]) for idx in used_z_indices] + [Z_IGNORE] * (LIB_SIZE - n_used)}
            for env in envs]
    return envs

def reorder_envs(f: Expr, envs: List[Dict]):
    z_indices = f.zs()
    sprite_indices = f.sprites()

    # reorder envs to place used indices at the front (to where they will be remapped)
    return [
        {'z': [util.unwrap_tensor(env['z'][i]) for i in z_indices] +
              [util.unwrap_tensor(env['z'][i]) for i in range(LIB_SIZE) if i not in z_indices],
         'sprites': [env['sprites'][i] for i in sprite_indices] +
                    [env['sprites'][i] for i in range(LIB_SIZE) if i not in sprite_indices],
        }
        for env in envs
    ]

def to_delta_examples(f: Expr, envs_choices: List[List[Dict]]):
    # simplifying envs wrt f is fine b/c p is a prefix of f
    envs_choices = [reorder_envs(f, envs) for envs in envs_choices]
    f = f.simplify_indices()
    f_toks = f.serialize()
    f_lines = f.lines()
    for f_envs, p_envs in it.product(envs_choices, envs_choices):
        f_bmps = T.stack([f.eval(env) for env in f_envs])
        for i in range(len(f_lines) + 1):
            p = Seq(*f_lines[:i])
            p_bmps = T.stack([p.eval(env) for env in p_envs])
            p_toks = p.serialize()
            d_toks = f_lines[i].serialize() if i < len(f_lines) else []
            yield (p_toks, p_envs, p_bmps), (f_toks, f_envs, f_bmps), d_toks

def gen_closures(n_envs: int, n_programs: int, n_lines: int,
                 arg_exprs: List[Expr],
                 n_zs: Tuple[int, int],
                 line_types: List[type], line_type_weights: List[float],
                 debug=False):
    """
    Generate `(f, z)` pairs with associated training outputs (value, policy).
    """
    for i in range(n_programs):
        envs = seed_envs(n_envs)
        p = make_flat_scene(n_objs=n_lines, envs=envs, height=B_H, width=B_W, include_zs=False, debug=False)
        # p = make_program(envs=envs, arg_exprs=arg_exprs, n_lines=n_lines, n_zs=n_zs,
        #                  line_types=line_types, line_type_weights=line_type_weights,
        #                  debug=debug)
        if debug: print(f'[{i}/{n_programs}]: {p}, {extract_zs(envs)}')
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

def is_valid(expr: Expr, envs: List[Dict]):
    for env in envs:
        try:
            expr.eval(env)
        except AssertionError:
            return False
    return True

def choose_random_sprite(envs: List[Dict], render_height: int, render_width: int):
    # Choose one of the sprites at random and position it in a random location
    sprite_index = random.randint(0, LIB_SIZE - 1)
    
    # Progressively bound the random location closer to the origin (0, 0) to
    # ensure that we eventually find a good place to put the sprite
    # -- this should terminate b/c all sprites are smaller than (B_H, B_W)
    height_bound, width_bound = render_height, render_width
    while height_bound >= 0 and width_bound >= 0:
        x, y = Num(random.randint(0, height_bound)), Num(random.randint(0, width_bound))
        if is_valid((sprite := Sprite(sprite_index, x, y)), envs):
            return sprite
        height_bound -= 1
        width_bound -= 1
    raise UnimplementedError("Shouldn't get here")

def make_rect(x, y, width, height, color=1):
    assert width >= 1 and height >= 1
    # assert x >= 0 and y >= 0
    return CornerRect(Num(x), Num(y), Num(x + width - 1), Num(y + height - 1), color=Num(color))

def roll_size(mu=3, sigma=2):
    return max(0, int(random.normalvariate(mu, sigma)))  # slower than random.gauss, but thread-safe

def random_z():
    return random.choice([Z(i) for i in range(LIB_SIZE)])

def max_dimensions(pos: Tuple[int, int], positions: List[Tuple[int, int]], height: int, width: int):
    # TODO: make this compute the frontier point that maximizes area
    px, py = pos
    nearest_x, nearest_y = width, height
    for x, y in positions:
        if px < x <= nearest_x and py < y <= nearest_y:
            nearest_x = x
            nearest_y = y
    return nearest_x - px, nearest_y - py

def plot_max_dimensions(positions, height, width):
    lines = [make_rect(x, y, 1, 1, color=1) for x, y in positions]
    for x, y in positions:
        w, h = max_dimensions((x, y), positions, height, width)
        box = make_rect(x, y, w, h, color=3)
        lines.append(box)
    viz.viz(Seq(*lines).eval(), title="Points with bounds")

def plot_points(points):
    viz.viz(Seq(*[make_rect(x, y, 1, 1, color=1) for x, y in points]).eval(),
            title="Points")

def dist(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

def dist_from_points(x, y, points: List[Tuple[int, int]]):
    return min(dist(x, y, px, py) for px, py in points)

def max_dist_point(points: List[Tuple[int, int]], height: int, width: int) -> Tuple[int, int]:
    """Returns the point on the grid furthest away from all other points"""
    return max(it.product(range(width), range(height)),
               key=lambda p: dist_from_points(p[0], p[1], points))

def uniform_random_points(n_points, gap, height, width, debug):
    """
    Pick some random points by choosing locations uniformly at random,
    but re-rolling a pick if it's too close to our prior picks.
    """
    points = []
    while len(points) < n_points:
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        if debug: print(points, x, y)
        if dist_from_points(x, y, points) >= gap:
            points.append((x, y))
    return points

def perturbed_grid_positions(n_points, height, width, debug):
    """
    Pick random points by placing points spaced out uniformly on the grid,
    then randomly perturb points
    """
    points = []
    pass
    
def max_space_positions(n_points, height, width, perturb, debug):
    points = [(random.randint(0, width-1),
               random.randint(0, height-1))]
    while len(points) < n_points:
        x, y = max_dist_point(points, height, width)
        if perturb:
            x = util.clamp(int(x + random.normalvariate(0, 1)), 0, width - 1)
            y = util.clamp(int(y + random.normalvariate(0, 1)), 0, height - 1)
        if debug: print(points, x, y)
        if dist_from_points(x, y, points) > 1:
            points.append((x, y))
    return points

def make_flat_scene(n_objs, envs, height, width, include_zs=False, debug=True):
    """Compose a program that generates rectangle/sprite scenes with minimal occlusion"""
    # seed a bunch of random positions and only keep the ones that have some distance between each other
    # positions = uniform_random_points(n_objs, 1, height, width, debug)
    positions = max_space_positions(n_objs, height, width, perturb=False, debug=debug)
    
    # choose random sizes that err towards being small
    lines = []
    for x, y in positions:
        if random.randint(0, 9) < 3:
            line = choose_random_sprite(envs, height-1, width-1)
        else:
            n_tries = 0
            while True:
                n_tries += 1
                if debug and n_tries % 100 == 0: print(f'[SRect]: {n_tries} tries')
                line = SizeRect(
                    Num(x) if not include_zs or random.randint(0, 4) > 0 else random_z(),
                    Num(y) if not include_zs or random.randint(0, 4) > 0 else random_z(),
                    Num(util.clamp(roll_size(2, 2), 1, width - x)),
                    Num(util.clamp(roll_size(2, 2), 1, height - y))
                )
                if is_valid(line, envs):
                    break
        
        # choose a relatively unused color to help variety
        line.color = choose_color(lines)
        lines.append(line)
        lines = canonical_ordering(lines)
        # lines = rm_dead_code(lines, envs, strict=True)
        
        if debug: print(lines)

    # favor putting random args as positions instead of dimensions
    return Seq(*lines)

def make_shape(shape_type: Type[Expr], envs: List[Dict],
               min_zs: int, max_zs: int,
               rand_exprs: List[Expr], const_exprs: List[Expr],
               debug=False):
    assert min_zs <= max_zs, f"min_zs ({min_zs}) > max_zs ({max_zs})"

    n_args = len(shape_type.in_types) - 1
    n_tries = 0
    while True:
        n_tries += 1
        if debug and n_tries % 1000 == 0:
            print(f'[{shape_type.__name__}]: {n_tries} tries')
        
        n_rand_args = random.randint(util.clamp(min_zs, 0, n_args),
                                     util.clamp(max_zs, 0, n_args))
        args = util.shuffled(random.choices(population=rand_exprs, k=n_rand_args) +
                             random.choices(population=const_exprs, k=n_args - n_rand_args))
        shape = shape_type(*args)
        
        if is_valid(shape, envs):
            return shape

def choose_color(lines: List[Expr]):
    # counts[i] = (the inverse of) the number of times Num(i-1) is used as a color in lines
    counts = [1 / (1 + sum(line.color == i for line in lines))
              for i in range(1, 10)]
    return Num(random.choices(population=range(1, 10), weights=counts, k=1)[0])

def make_program(envs: List[dict], arg_exprs: List[Expr],
                 n_lines: int, n_zs: Tuple[int, int],
                 line_types: List[type], line_type_weights: List[float] = None,
                 render_height=B_H, render_width=B_W,
                 debug=False):
    """
    Creates a new program matching the spec. Uses expressions containing randomness for arguments
    of each line object, but doesn't allow random colors.
    """
    rand_exprs, const_exprs = util.split(arg_exprs, lambda expr: expr.zs())
    lines = []
    while len(lines) < n_lines:
        is_sprite = random.choices(population=[True, False], weights=[1, len(line_types) * 2], k=1)[0]
        if is_sprite:
            line = choose_random_sprite(envs, render_height - 1, render_width - 1)
        else:
            shape_type = random.choices(population=line_types, weights=line_type_weights)[0]
            line = make_shape(
                shape_type=shape_type, envs=envs,
                min_zs=n_zs[0], max_zs=n_zs[1],
                rand_exprs=rand_exprs, const_exprs=const_exprs,
                debug=debug
            )

        # choose a random color; favor colors not often used
        line.color = choose_color(lines)

        if debug: print(lines)
        lines.append(line)
        lines = canonical_ordering(lines)
        lines = rm_dead_code(lines, envs, strict=True)

        if debug:
            program = Seq(*lines)
            viz.viz_mult(T.stack([program.eval(env) for env in envs]), text=program)
    
    return Seq(*lines)

def rm_dead_code(lines: List[Expr], envs, strict=False):
    """
    Remove any lines that don't affect the render of Seq(lines) in any of the provided environments.
    
    Exploits the fact that earlier lines in the sequence are rendered first.
    """
    orig_bmps = render(Seq(*lines), envs)
    out = []
    for i, line in enumerate(lines):
        excl_lines = out + lines[i + 1:]
        excl_bmps = render(Seq(*excl_lines), envs)
        if (not strict and not T.equal(excl_bmps, orig_bmps)) or \
           (strict and not any(T.equal(x, y) for x, y in zip(excl_bmps, orig_bmps))):
            out.append(line)
    return out

def canonical_ordering(lines: List[Expr]):
    def destructure(e):
        if isinstance(e, Point):
            return 0, e.x, e.y
        elif isinstance(e, Line):
            return 1, e.x1, e.y1, e.x2, e.y2
        elif isinstance(e, CornerRect):
            return 2, e.x_min, e.y_min, e.x_max, e.y_max, e.color
        elif isinstance(e, SizeRect):
            return 3, e.x, e.y, e.w, e.h, e.color
        if isinstance(e, Sprite):
            return 4, e.i, e.x, e.y
    
    return sorted(lines, key=destructure)

def extract_zs(envs):
    return [env['z'] for env in envs]

def demo_create_program():
    envs = seed_envs(5)
    p = make_program(envs=envs,
                     arg_exprs=[Z(i) for i in range(LIB_SIZE)] + [Num(i) for i in range(0, 10)],
                     n_lines=3,
                     line_types=[CornerRect, Line, Point],
                     line_type_weights=[4, 3, 1],
                     n_zs=(0, 2),
                     debug=True)
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
                       arg_exprs, (0, 2), [CornerRect, Line, Point], [4, 3, 1], True)
                      for i in range(workload_sz)])

    for i, (f, envs) in enumerate(util.load_incremental(f'{fname}*.dat')):
        print(i, f)

def gen_closures_and_deltas(worker_id: int, closures_loc: str, deltas_loc: str,
                            n_envs: int, n_programs: int, n_lines: int,
                            arg_exprs: List[Expr],
                            n_zs: Tuple[int, int],
                            line_types: List[type], line_type_weights: List[float],
                            hetero_zs=False,
                            debug=False):
    util.make_parent_dir(closures_loc)
    util.make_parent_dir(deltas_loc)
    with open(closures_loc, 'wb') as closures_file, open(deltas_loc, 'wb') as deltas_file:
        gen = gen_closures(
            n_envs=n_envs * (1 + int(hetero_zs)),
            n_programs=n_programs,
            n_lines=n_lines,
            arg_exprs=arg_exprs,
            n_zs=n_zs,
            line_types=line_types,
            line_type_weights=line_type_weights,
            debug=debug
        )
        for i, (f, envs) in enumerate(gen):
            pickle.dump((f, envs), closures_file)
            printed = False
            if hetero_zs:
                env_sets = [envs[:n_envs], envs[n_envs:]]
            else:
                env_sets = [envs]
            for (p_toks, p_envs, p_bmps), (f_toks, f_envs, f_bmps), d_toks in to_delta_examples(f, env_sets):
                pickle.dump(
                    ((p_toks, p_bmps), (f_toks, f_bmps), d_toks),  # ignore envs
                    deltas_file
                )
                if not printed and debug:
                    print(f'[{worker_id}][{i}/{n_programs}]: {f_toks}')
                    # p = deserialize(f_toks)
                    # if p.zs() and n_lines >= 2:
                    #     viz.viz_mult([p.eval(env) for env in f_envs])
                    printed = True
                print(f'[{worker_id}][{i}/{n_programs}]:', p_toks, d_toks, f_toks, sep='\n', end='\n\n')

def gen_closures_and_deltas_mp(closures_loc_prefix: str, deltas_loc_prefix: str,
                               n_envs: int, n_programs: int, n_lines_bounds: Tuple[int, int], n_zs: Tuple[int, int],
                               line_types: List[type], line_type_weights: Optional[List[float]] = None,
                               hetero_zs=False,
                               debug=False,
                               n_workers: int = 1):
    """
    Generate a stream of closures and directly convert them to deltas. Write both to files.
    """
    assert n_programs % n_workers == 0, f'Expected n_workers to divide n_programs, got {n_programs}/{n_workers}'
    n_lines_lo, n_lines_hi = n_lines_bounds
    assert n_lines_lo >= 0, f'Expected n_lines_bounds > 1, found {n_lines_bounds}'
    
    # run n_workers workers to generate a total of n_programs programs of each size in n_lines
    arg_exprs = (
            # [XMax(), YMax()] +
            [Z(i) for i in range(LIB_SIZE)] +
            [Num(i) for i in range(Z_LO, Z_HI+1)]
    )
    n_programs_per_worker = n_programs // n_workers
    with mp.Pool(processes=n_workers) as pool:
        pool.starmap(gen_closures_and_deltas,
                     [(i,
                       f'{closures_loc_prefix}closures_{i}.dat',
                       f'{deltas_loc_prefix}deltas_{i}.dat',
                       n_envs, n_programs_per_worker, n_lines,
                       arg_exprs, n_zs, line_types, line_type_weights,
                       hetero_zs,
                       debug)
                      for n_lines in range(n_lines_lo, n_lines_hi + 1)
                      for i in range(n_workers)])
    # separate closure and delta gen? might allow better allocation of workers

def tb_viz_data(dataset: Iterable):
    """Use Tensorbaord to visualize the data in a dataset"""
    writer = tb.SummaryWriter(comment='_dataviz')
    for i, ((p, z_p, b_p), (f, z_f, b_f), d) in enumerate(dataset):
        writer.add_text('p-d-f', f'p={p}\n'
                                 f'd={d}\n'
                                 f'f={f}', i)
        writer.add_images('b_f', b_f.unsqueeze(1), i)
        writer.add_images('b_p', b_p.unsqueeze(1), i)

def viz_data(dataset: Iterable):
    for i, ((p, z_p, b_p), (f, z_f, b_f), d) in enumerate(dataset):
        print('p:', p)
        print('d:', d)
        print('f:', f)
        viz.viz_mult(b_f, f)
        print()

# Examine datasets
def collect_stats(dataset: Iterable, max_line_count=3):
    by_len = {
        i: {"count": 0,    # number of programs of length i
            "overlap": 0}  # overlap between rendered lines
        for i in range(1, max_line_count+1)
    }
    n_of_type = {t: 0 for t in [Point, Line, CornerRect]}  # track number of lines by type
    seen_lines = set()  # track unique lines

    n_items = 0
    total_n_lines = 0
    seen_fs = set()
    i = 0
    for (p, z_p, b_p), (f, z_f, b_f), d in dataset:
        i += 1
        if i > 10_000: break
        # print(i, f)
        if str(f) in seen_fs:
            continue
        else:
            seen_fs.add(str(f))
        
        n_items += 1
        f = deserialize(f)
        lines = f.lines()
        n_lines = len(lines)
        total_n_lines += n_lines
        
        z_f, z_p = z_f[0], z_p[0]  # FIXME

        seen_lines = seen_lines.union(lines)
        by_len[n_lines]["count"] += 1

        for line in lines:
            t = type(line)
            n_of_type[t] += 1

        overlap_p = T.sum(T.sum(T.stack([line.eval(z_p) > 0 for line in lines]), dim=0) > 1).item()
        overlap_f = T.sum(T.sum(T.stack([line.eval(z_f) > 0 for line in lines]), dim=0) > 1).item()
        
        by_len[n_lines]["overlap"] = (overlap_p + overlap_f)/2
    
    print(f"#items: {n_items}",
          f"Unique lines: {len(seen_lines)}, (expected: {total_n_lines})",
          f"Number of lines by type: {n_of_type}",
          f"Counts by length:",
          *[f"  {i}: {by_len[i]['count']}" for i in range(1, max_line_count+1)],
          f"Overlaps by length:",
          *[f"  {i}: {by_len[i]['overlap']}" for i in range(1, max_line_count+1)],
          sep="\n")

def test_reorder_envs():
    cases = [
        (Seq(CornerRect(Z(7), Z(2), Z(9), Z(10)), Sprite(2), Sprite(1)),
         [{'z': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
           'sprites': [util.img_to_tensor(["###",
                                           "#_#",
                                           "###"]),
                       util.img_to_tensor(["##",
                                           "_#",
                                           "_#"]),
                       util.img_to_tensor(["###",
                                           "_#_",
                                           "###"])]},
          ],
         [{'z': [7, 2, 9, 10, 0, 1, 3, 4, 5, 6, 8],
           'sprites': [
               util.img_to_tensor(["###",
                                   "_#_",
                                   "###"]),
               util.img_to_tensor(["##",
                                   "_#",
                                   "_#"]),
               util.img_to_tensor(["###",
                                   "#_#",
                                   "###"]),
           ]},
          ]),
    ]
    
    def equal(a: List[Dict], b: List[Dict]) -> bool:
        for e1, e2 in zip(a, b):
            if e1['z'] != e2['z']: return False
            for s1, s2 in zip(e1['sprites'], e2['sprites']):
                if not T.equal(s1, s2): return False
        return True
        
    for f, envs, expected in cases:
        out = reorder_envs(f, envs)
        assert equal(expected, out), f'Expected={expected}, but got {out}'
    print(" [+] passed test_reorder_envs")

def demo_flat_scenes():
    for i in range(10):
        envs = seed_envs(3)
        p = make_flat_scene(n_objs=10, width=B_W, height=B_H, envs=envs, include_zs=False, debug=True)
        renders = [p.eval(env) for env in envs]
        viz.viz_mult(renders, text=p)

def demo_positions():
    for i in range(10):
        points = max_space_positions(30, B_H, B_W, perturb=False, debug=True)
        plot_points(points)


if __name__ == '__main__':
    # demo_gen_program()
    # demo_gen_closures()
    # demo_gen_policy_data()
    # test_reorder_envs()

    demo_flat_scenes()
    exit(0)

    if len(sys.argv) - 1 != 6:
        print("Usage: data.py dir mode min_zs max_zs n_lines t")
        exit(1)
    
    dir, mode, min_zs, max_zs, n_lines, t = sys.argv[1:]
    min_zs = int(min_zs)
    max_zs = int(max_zs)
    n_lines = int(n_lines)
    n_envs = 5
    n_programs = 10
    s_n_programs = '10'
    n_workers = 1
    z_code = f'{min_zs}~{max_zs}' if min_zs < max_zs else f'{min_zs}'
    code = f'{s_n_programs}-R-{n_envs}e{n_lines}l{z_code}z'

    print(f"Generating policy data for code={code}, mode={mode}, at time {t} with dir={dir}")
    gen_closures_and_deltas_mp(
        closures_loc_prefix=f'{dir}/{code}/{t}/{mode}/',
        deltas_loc_prefix=f'{dir}/{code}/{t}/{mode}/',
        n_envs=n_envs,
        n_programs=n_programs,
        n_lines_bounds=(n_lines, n_lines),
        n_zs=(min_zs, max_zs),
        line_types=[CornerRect],
        line_type_weights=[1],
        n_workers=n_workers,
        hetero_zs=False,
        debug=True,
    )

    # viz_data(util.load_incremental('../data/policy-pretraining/10-R-5e1l0~1z/'
    #                                f'{t}/training/deltas_*.dat'))

    # for n_lines in line_range:
    #     code = f'{s_n_programs}-R-{n_envs}e{n_lines}l{z_code}z'
    #     for mode in ['training', 'validation']:
    #         print(f"Joining for code={code}, mode={mode}")
    #         util.join_glob(f"{dir}/{code}/{t}/{mode}/deltas_*.dat",
    #                        f"{dir}/{code}/{t}/{mode}_deltas.dat")

    # for mode in ['training', 'validation']:
    #     print(f"Weaving across line numbers for mode={mode}...")
    #     prefix = f'{dir}/{s_n_programs}-R-{n_envs}e'
    #     line_code = f'{min(line_range)}~{max(line_range)}'
    #     util.weave_glob(f"{prefix}*l{z_code}z/{t}/{mode}_deltas.dat",
    #                     f"{prefix}{line_code}l{z_code}z/{t}/{mode}_deltas_weave.dat")
    
