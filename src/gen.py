"""
Make programs to use as training data
"""

import pdb
from math import floor, sqrt
from random import choice, randint, shuffle
import multiprocessing as mp

from viz import viz_grid
from grammar import *
from bottom_up import bottom_up_generator, eval


def gen_shapes(n_shapes, a_exprs, envs, shape_types, min_zs=0, max_zs=None):
    z_exprs, c_exprs = util.split(a_exprs, lambda a: a.zs())
    if max_zs is None: max_zs = len(z_exprs)
    if min_zs is None: min_zs = max_zs
    assert max_zs <= len(z_exprs), f'Expected max_zs <= |zs|, but found max_zs={max_zs}, |zs|={len(z_exprs)}'
    assert min_zs <= max_zs, f'Expected min_zs <= max_zs, but found min_zs={min_zs}, max_zs={max_zs}'

    shapes = set()
    for shape_type in shape_types:
        n_tries = 0
        n_hits = 0
        while n_hits < n_shapes:
            color = Num(randint(1, 9))
            n_zs = min(len(shape_type.in_types), randint(min_zs, max_zs)) # cap n_zs by number of args for shape type
            z_args = [choice(z_exprs) for _ in shape_type.in_types[:n_zs]]
            c_args = [choice(c_exprs) for _ in shape_type.in_types[n_zs:]]
            args = z_args + c_args
            shuffle(args)
            shape = shape_type(*args[:-1], color)
            if shape not in shapes and all(out is not None for out in eval(shape, envs)):
                shapes.add(shape)
                n_hits += 1
                print(f'{shape_type} hits: {n_hits}/{n_tries}')
            n_tries += 1
        print(f'{shape_type} hits: {n_hits}/{n_tries}')
    return shapes

def gen_shapes_mp(n_shapes, a_exprs, envs, shape_types, min_zs=0, max_zs=None, n_processes=1):
    with mp.Pool(n_processes) as pool:
        sets = pool.starmap(gen_shapes, [(chunk_size,
                                          a_exprs,
                                          envs,
                                          shape_types,
                                          min_zs,
                                          max_zs)
                                         for chunk_size in util.chunk(n_shapes, n_processes)])
    return set.union(*sets)

def enum_shapes(envs, shape_types, max_zs=0):
    assert max_zs <= LIB_SIZE
    scene_gram = Grammar(ops=shape_types,
                         consts=([Z(i) for i in range(max_zs)] +
                                 [Num(i) for i in range(0, 10)]))
    shapes = set()
    for shape_type in shape_types:
        expr_sz = len(shape_type.in_types) + 1
        for shape, sz in bottom_up_generator(expr_sz, scene_gram, envs):
            if all(out is not None for out in eval(shape, envs)):
                shapes.add(shape)
    print(f'Enumerated {len(shapes)} shapes.')
    return shapes

def rm_dead_code(shapes, envs):
    """
    Remove any occluded shapes in the scene w.r.t. envs
    """
    def bmps(xs):
        return T.stack([Seq(*xs).eval(env) for env in envs])

    orig_bmps = bmps(shapes)
    keep = []
    for i, shape in enumerate(shapes):
        excl_bmps = bmps(keep + shapes[i+1:])
        if (excl_bmps != orig_bmps).any():
            keep.append(shape)
    return keep

def canonical_ordering(scene):
    """
    Order entities by complexity and position.
    - Complexity: Point < Line < Rect
    - Positions are sorted in y-major order (e.g., (1, 4) < (2, 1) and (3, 1) < (3, 2))
    """
    def destructure(e):
        if isinstance(e, Point):
            return 0, e.x, e.y
        elif isinstance(e, Line):
            return 1, e.x1, e.y1, e.x2, e.y2
        elif isinstance(e, Rect):
            return 2, e.x, e.y, e.w, e.h

    return sorted(scene, key=destructure)

def gen_random_scene(shapes, envs, n_shapes):
    """
    Generate a random sequence of shapes.
    - dead code removal
    - canonical ordering
    """
    # TODO: transforms
    # TODO: don't sample elements of `shapes` that have previously been used
    scene = []
    while len(scene) < n_shapes:
        shape = choice(shapes)
        scene.append(shape)
        scene = canonical_ordering(scene) # edit ordering before checking for overlaps
        scene = rm_dead_code(scene, envs)
    return Seq(*scene)

def gen_exs(n_exs, n_shapes, shapes, envs, save_to, verbose=True):
    """
    Generates a set of (bmp set, program) pairs from an expression generator on n entities.
    Writes the set in a stream to the file `{fname}_{n_shapes}.exs`
    """
    fname = f'{save_to}_{n_shapes}.exs'
    cleared = False
    for i in range(n_exs):
        scene = gen_random_scene(shapes, envs, n_shapes)
        if verbose: print(f'scene generated [{n_shapes}][{i+1}/{n_exs}]: {scene}')
        bmps = [scene.eval(env) for env in envs]
        prog = scene.simplify_indices().serialize()
        util.save((bmps, prog), fname=fname, append=cleared, verbose=False)
        cleared = True

def gen_exs_mp(n_exs, shapes, envs, min_shapes, max_shapes, save_to):
    n_sizes = max_shapes - min_shapes + 1
    shapes = list(shapes)
    with mp.Pool(n_sizes) as pool:
        pool.starmap(gen_exs, [(n_exs,
                                i,
                                shapes,
                                envs,
                                save_to)
                               for i in range(min_shapes, max_shapes+1)])

def rand_sprite(envs, a_exprs, i=-1, color=-1):
    i = i if 0 <= i < LIB_SIZE else randrange(1, LIB_SIZE)
    color = color if isinstance(color, Num) and 1 <= color.n <= Z_HI else Num(randint(1, Z_HI))
    n_misses = 0
    while True:
        s = Sprite(i, choice(a_exprs), choice(a_exprs))
        outs = eval(s, envs)
        if all(out is not None for out in outs):
            break
        else:
            n_misses += 1
    print(f'sprite misses:', n_misses)
    return Apply(Recolor(color), s) if color > 1 else s

def rand_transform(e):
    """
    Return a random (nontrivial) transformation of e
    """
    # # Transform
    # n = choice(a_exprs)
    # t = choice(transforms)
    # t_args = [choice(a_exprs) for _ in t.in_types]
    # if any(n.eval(env) > 0 for env in envs):
    #     t = Repeat(t(*t_args), n)
    # else:
    #     t = t(*t_args)
    assert False, "not implemented"

def viz_sprites(envs):
    k = floor(sqrt(LIB_SIZE))
    for env in envs:
        viz_grid(env['sprites'][:k**2], text=f'sprites[:{k**2}]')

def test_canonical_ordering():
    test_cases = [
        ([
            Point(Num(0), Num(1)),
            Point(Num(1), Num(0))
         ],
         [
             Point(Num(0), Num(1)),
             Point(Num(1), Num(0))
         ]),
        ([
            Point(Num(2), Num(1)),
            Point(Num(2), Num(0))
         ],
         [
             Point(Num(2), Num(0)),
             Point(Num(2), Num(1))
         ]),
        ([
            Line(Num(2), Num(0), Num(3), Num(1)),
            Rect(Num(1), Num(0), Num(1), Num(1)),
            Point(Num(2), Num(1)),
            Rect(Num(0), Num(1), Num(2), Num(2)),
            Rect(Num(0), Num(1), Num(3), Num(2)),
            Point(Num(0), Num(2)),
            Line(Num(0), Num(0), Num(3), Num(3)),
         ],
         [
            Point(Num(0), Num(2)),
            Point(Num(2), Num(1)),
            Line(Num(0), Num(0), Num(3), Num(3)),
            Line(Num(2), Num(0), Num(3), Num(1)),
            Rect(Num(0), Num(1), Num(2), Num(2)),
            Rect(Num(0), Num(1), Num(3), Num(2)),
            Rect(Num(1), Num(0), Num(1), Num(1)),
         ]),
    ]
    for entities, ans in test_cases:
        out = canonical_ordering(entities)
        assert out == ans, f'Failed test: expected {ans}, got {out}'
    print("[+] passed test_canonical_ordering")

def test_rm_dead_code():
    zs = [
        [0, 1],
        [0, 2],
    ]
    test_cases = [
        ([
            ["1_",
             "1_"],
        ],
         [Sprite(0)],
         [Sprite(0)]),
        ([
            ["1_",
             "1_"],
            ["22",
             "22"],
          ],
         [Sprite(0), Sprite(1)],
         [Sprite(0), Sprite(1)]),
        ([
            ["1_",
             "1_"],
            ["2_",
             "2_"],
          ],
         [Sprite(0), Sprite(1)],
         [Sprite(0)]),
        ([
            ["11",
             "1_"],
            ["__",
             "_2"],
            ["33",
             "33"],
          ],
         [Sprite(0), Sprite(1), Sprite(2)],
         [Sprite(0), Sprite(1)]),
        ([],
         [Rect(Num(0), Num(0), Num(2), Num(2)),
          Rect(Num(0), Num(0), Num(2), Num(2))],
         [Rect(Num(0), Num(0), Num(2), Num(2))]),
        ([],
         [Rect(Num(0), Num(0), Num(2), Num(2)),
          Line(Num(0), Num(0), Num(1), Num(1))],
         [Rect(Num(0), Num(0), Num(2), Num(2))]),
        ([],
         [Line(Num(0), Num(0), Num(1), Num(1), color=Num(1)),
          Rect(Num(0), Num(0), Num(2), Num(2), color=Num(1))],
         [Rect(Num(0), Num(0), Num(2), Num(2), color=Num(1))]),
        ([],
         [Line(Num(0), Num(0), Num(1), Num(1), color=Num(1)),
          Rect(Num(0), Num(0), Num(2), Num(2), color=Num(2))],
         [Line(Num(0), Num(0), Num(1), Num(1), color=Num(1)),
          Rect(Num(0), Num(0), Num(2), Num(2), color=Num(2))]),
        ([],
         [Rect(Z(0), Z(0), Num(2), Num(2)), Point(Z(0), Plus(Z(0), Num(1)))],
         [Rect(Z(0), Z(0), Num(2), Num(2))]),
        ([],
         [Rect(Z(0), Z(1), Num(2), Num(2)), Point(Z(0), Z(1))],
         [Rect(Z(0), Z(1), Num(2), Num(2))]),
        ([],
         [Rect(Z(0), Z(0), Num(2), Num(2)), Point(Z(0), Plus(Z(1), Num(1)))],
         [Rect(Z(0), Z(0), Num(2), Num(2)), Point(Z(0), Plus(Z(1), Num(1)))]),
    ]
    for lib, entities, ans in test_cases:
        lib = [util.img_to_tensor(b, w=B_W, h=B_H) for b in lib]
        envs = [{'z': z, 'sprites': lib} for z in zs]
        out = rm_dead_code(entities, envs)
        assert out == ans, f"Failed test: in={entities}; expected {ans}, got {out}"
    print("[+] passed test_rm_dead_code")

def load_shapes(fname):
    assert False, 'unimplemented'

def make_exs(n_exs,             # number of total (bitmaps, program) pairs to make
             n_envs,            # number of envs (bitmaps) per program
             min_shapes,        # min number of shapes in each scene
             max_shapes,        # max number of shapes in each scene
             a_grammar,         # grammar for arithmetic exprs (components of shapes)
             a_bound,           # bound on arithmetic exprs in each entity
             shape_types,       # shape types allowed
             fname,             # prefix to use for filenames
             enum_all_shapes=False,  # whether to randomly generate shapes or enumerate all possibilities
             label_zs=True,     # whether to label z's with indices or not (e.g. just have 'z' instead of 'z_0') TODO
             min_zs=0,          # min number of z's to allow in each entity
             max_zs=None,       # max number of z's to allow in each entity
             n_processes=1):
    print(f'Parameters: n_exs={n_exs}, n_envs={n_envs}, zs=({min_zs}, {max_zs}), ' 
          f'shape_types={shape_types}, min_shapes={min_shapes}, max_shapes={max_shapes}, ' 
          f'a_grammar={a_grammar}, a_bound={a_bound}, shape_types={shape_types}')

    # Generate and save shapes
    envs = [{'z': seed_zs(), 'sprites': seed_sprites()} for _ in range(n_envs)]
    a_exprs = [a_expr for a_expr, size in bottom_up_generator(a_bound, a_grammar, envs)]
    n_shapes = n_exs * max_shapes
    if enum_all_shapes:
        shapes = enum_shapes(envs, shape_types, max_zs)
    else:
        shapes = gen_shapes_mp(n_shapes, a_exprs, envs, shape_types, min_zs, max_zs, n_processes)
    util.save({'n_shapes': n_shapes,
               'shape_types': shape_types,
               'shapes': shapes,
               'zs': (min_zs, max_zs),
               'n_envs': n_envs,
               'envs': envs,
               'a_bound': a_bound,
               'a_exprs': a_exprs},
              f'{fname}.cmps',
              append=False)

    exit(0)

    # Generate and save exprs w/ bmp outputs
    gen_exs_mp(n_exs=n_exs, shapes=shapes, envs=envs,
               min_shapes=min_shapes, max_shapes=max_shapes, save_to=fname)


def viz_exs(fname):
    for bmps, tokens in util.load_incremental(fname):
        print('tokens:', tokens)
        expr = deserialize(tokens)
        viz_grid(bmps[:9], expr)

def list_cmps(fname):
    cmps = util.load(fname)
    for shape in cmps['shapes']:
        print(shape)

def list_exs(fname):
    for bmps, tokens in util.load_incremental(fname):
        print(deserialize(tokens), f'{len(bmps)} bitmaps')

def count_uniq(fname):
    d = {}
    for bmps, tokens in util.load_incremental(fname):
        for e in deserialize(tokens).bmps:
            d[e] = d.get(e, 0) + 1
    return d

def compare(f1, f2):
    d1 = count_uniq(f1)
    d2 = count_uniq(f2)
    n_keys = (len(d1) + len(d2))//2
    val_capacity = sum(d1.values()) + sum(d2.values())

    key_overlap = 0
    abs_overlap = 0
    for e in d1.keys():
        if e in d2:
            print(f'overlap: {e}')
            key_overlap += 1
            abs_overlap += min(d1[e], d2[e])
    return key_overlap, abs_overlap, n_keys, val_capacity

def shape_code(shape_types):
    return f"{'p' if Point in shape_types else ''}" \
           f"{'l' if Line in shape_types else ''}" \
           f"{'r' if Rect in shape_types else ''}"

def run_cfgs(cfgs):
    for cfg in cfgs:
        sc = shape_code(cfg['shape_types'])
        code = f"{cfg['n_exs']}-{cfg['min_shapes']}~{cfg['max_shapes']}{sc}{cfg['max_zs']}z{cfg['n_envs']}e"
        print(f'code: {code}')

        for mode in ['train', 'test']:
            make_exs(n_exs=cfg['n_exs'],
                     shape_types=cfg['shape_types'],
                     min_shapes=cfg['min_shapes'],
                     max_shapes=cfg['max_shapes'],
                     n_envs=cfg['n_envs'],
                     max_zs=cfg['max_zs'],
                     min_zs=cfg['min_zs'],
                     a_bound=cfg['a_bound'],
                     a_grammar=cfg['a_grammar'],
                     fname=f'../data/{code}-{mode}',
                     enum_all_shapes=cfg['enum_all_shapes'],
                     n_processes=cfg['n_processes'],
                     label_zs=cfg['label_zs'])


if __name__ == '__main__':
    # test_rm_dead_code()
    # test_canonical_ordering()

    a_grammar = Grammar(ops=[Plus, Minus, Times],
                        consts=([Z(i) for i in range(LIB_SIZE)] +
                                [Num(i) for i in range(0, 10)]))
    print(a_grammar.ops, a_grammar.consts)
    cfgs = [
        {
            'n_exs': 1_000_000,
            'shape_types': [Rect],
            'enum_all_shapes': True,
            'min_shapes': 1,
            'max_shapes': 5,
            'max_zs': 1,
            'min_zs': 0,
            'a_bound': 1,
            'n_envs': 5,
            'label_zs': True,
            'n_processes': 1,
            'a_grammar': a_grammar,
        },
    ]
    run_cfgs(cfgs)
    # list_cmps(f"../data/1000000-1~5r1z5e-train.cmps")
