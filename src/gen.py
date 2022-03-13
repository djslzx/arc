"""
Make programs to use as training data
"""

import pdb
from math import floor, sqrt
from random import choice, randint, shuffle
import time
import multiprocessing as mp
import itertools as it

import viz
from grammar import *
from bottom_up import bottom_up_generator, eval
from transformer import recover_model

def gen_shapes(n_shapes, a_exprs, envs, shape_types, min_zs=0, max_zs=None):
    z_exprs, c_exprs = util.split(a_exprs, lambda a: a.zs())
    if max_zs is None: max_zs = len(z_exprs)
    if min_zs is None: min_zs = max_zs
    assert max_zs <= len(z_exprs), f'Expected max_zs <= |zs|, but found max_zs={max_zs}, |zs|={len(z_exprs)}'
    assert min_zs <= max_zs, f'Expected min_zs <= max_zs, but found min_zs={min_zs}, max_zs={max_zs}'

    def is_blank(bmp):
        return (bmp == 0).all()

    shapes = set()
    for shape_type in shape_types:
        n_tries = 0
        n_hits = 0
        while n_hits < n_shapes:
            color = Num(randint(1, 9))
            n_zs = min(len(shape_type.in_types), randint(min_zs, max_zs))  # cap n_zs by number of args for shape type
            z_args = [choice(z_exprs) for _ in shape_type.in_types[:n_zs]]
            c_args = [choice(c_exprs) for _ in shape_type.in_types[n_zs:]]
            args = z_args + c_args
            shuffle(args)
            shape = shape_type(*args[:-1], color)
            if shape not in shapes and all(out is not None and not is_blank(out)
                                           for out in eval(shape, envs)):
                shapes.add(shape)
                n_hits += 1
                if n_hits % 100 == 0:
                    print(f'{shape_type} hits: {n_hits}/{n_tries}')
            if n_tries % 10000 == 0:
                print(f'{shape_type} hits: {n_hits}/{n_tries}')
            n_tries += 1
        print(f'{shape_type} hits: {n_hits}/{n_tries}')
    return shapes

def gen_shapes_mp(n_shapes, a_exprs, envs, shape_types, min_zs=0, max_zs=None, n_processes=1):
    with mp.Pool(n_processes) as pool:
        sets = pool.starmap(gen_shapes, [(chunk_size, a_exprs, envs, shape_types, min_zs, max_zs)
                                         for chunk_size in util.chunk(n_shapes, n_processes)])
    return set.union(*sets)

def enum_shapes(envs, shape_types, min_zs=0, max_zs=0):
    assert max_zs <= LIB_SIZE
    scene_gram = Grammar(ops=shape_types,
                         consts=([Z(i) for i in range(min_zs, max_zs)] +
                                 [Num(i) for i in range(0, 10)]))
    shapes = set()
    for shape_type in shape_types:
        expr_sz = len(shape_type.in_types) + 1
        for expr, sz in bottom_up_generator(expr_sz, scene_gram, envs):
            if isinstance(expr, shape_type) and all(out is not None for out in eval(expr, envs)):
                shapes.add(expr)
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
            return 2, e.x_min, e.y_min, e.x_max, e.y_max

    return sorted(scene, key=destructure)

def gen_random_scene(shapes, envs, n_shapes):
    """
    Generate a random sequence of shapes in canonical order with dead code removal.
    """
    # TODO: transforms
    # TODO: don't sample elements of `shapes` that have previously been used
    scene = []
    while len(scene) < n_shapes:
        shape = choice(shapes)
        scene.append(shape)
        scene = canonical_ordering(scene)  # edit ordering before checking for overlaps
        scene = rm_dead_code(scene, envs)
    return scene

def gen_tf_exs(n_exs, n_shapes, shapes, envs, save_dir, verbose=True):
    """
    Generates a set of (bmp set, program) pairs.
    """
    t = time.time()
    fname = f'{save_dir}/{n_shapes}_{t}.tf.exs'
    cleared = False
    for i in range(n_exs):
        scene = Seq(*gen_random_scene(shapes, envs, n_shapes))
        if verbose: print(f'scene generated [{n_shapes}][{i+1}/{n_exs}]: {scene}')
        bmps = [scene.eval(env) for env in envs]
        prog = scene.simplify_indices().serialize()
        util.save((bmps, prog), fname=fname, append=cleared, verbose=False)
        cleared = True

def gen_tf_exs_mp(n_exs, shapes, envs, min_shapes, max_shapes, save_dir, n_processes=1):
    n_sizes = max_shapes - min_shapes + 1
    shapes = list(shapes)
    with mp.Pool(n_processes * n_sizes) as pool:
        pool.starmap(gen_tf_exs, [(n_scenes, n_shapes, shapes, envs, save_dir)
                                  for n_scenes, n_shapes in
                                  zip(util.chunk(n_exs, n_sizes * n_processes),
                                      it.chain.from_iterable([k] * n_processes
                                                             for k in range(min_shapes, max_shapes+1)))])

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
    
def make_shapes(shape_types, envs, a_exprs, n_zs, scene_sizes, save_dir,
                n_shapes=None, enum=False, n_processes=1):
    min_zs, max_zs = n_zs
    n_envs = len(envs)
    if enum:
        print("Enumerating all shapes...")
        shapes = enum_shapes(envs, shape_types, min_zs, max_zs)
        n_shapes = len(shapes)
        print(f"Enumerated {n_shapes} shapes.")
    else:
        print(f"Generating {n_shapes} shapes...")
        shapes = gen_shapes_mp(n_shapes, a_exprs, envs, shape_types, min_zs, max_zs, n_processes)
    util.save({'n_shapes': n_shapes,
               'shape_types': shape_types,
               'shapes': shapes,
               'zs': (min_zs, max_zs),
               'sizes': scene_sizes,
               'n_envs': n_envs,
               'envs': envs,
               'a_exprs': a_exprs},
              f'{save_dir}/{n_shapes}n{scene_sizes[0]}~{scene_sizes[1]}s{min_zs}~{max_zs}z{n_envs}e.cmps',
              append=False)
    return shapes

def make_tf_exs(n_exs, n_envs, shape_types, scene_sizes, a_grammar, a_depth,
                dir_name=None, n_zs=(0, 0), enum_all_shapes=False, label_zs=True, n_processes=1):
    if dir_name is None:
        code = make_name_code(n_exs=n_exs, scene_sizes=scene_sizes, shape_types=shape_types, n_zs=n_zs, n_envs=n_envs)
        dir_name = f'../data/{code}-tf'
    
    min_shapes, max_shapes = scene_sizes
    min_zs, max_zs = n_zs
    print(f'Parameters: n_exs={n_exs}, n_envs={n_envs}, zs=({min_zs}, {max_zs}), '
          f'shape_types={shape_types}, min_shapes={min_shapes}, max_shapes={max_shapes}, ' 
          f'a_grammar={a_grammar}, a_depth={a_depth}')

    # Generate and save shapes
    envs = [{'z': seed_zs(), 'sprites': seed_sprites()} for _ in range(n_envs)]
    a_exprs = [a_expr for a_expr, size in bottom_up_generator(a_depth, a_grammar, envs)]
    shapes = make_shapes(shape_types=shape_types, envs=envs, a_exprs=a_exprs, n_zs=n_zs, scene_sizes=scene_sizes,
                         save_dir=f'{dir_name}', n_shapes=n_exs * max_shapes, enum=enum_all_shapes,
                         n_processes=n_processes)

    # Generate and save exprs w/ bmp outputs
    gen_tf_exs_mp(n_exs=n_exs, shapes=shapes, envs=envs,
                  min_shapes=min_shapes, max_shapes=max_shapes, save_dir=dir_name)

def make_discrim_exs_combi(n_exs, n_envs, shape_types, scene_sizes, a_grammar, a_depth,
                           dir_name=None, n_zs=(0, 0), enum_all_shapes=False, n_processes=1):
    """
    Generate training examples for the discriminator.
    
    Examples take the form ((B1, B2), Y) where Y.
    Let P1 and P2 be the programs generating B1 and B2, respectively.
    Y's components are as follows:
     - a 0/1 value: 1 if P1 == P2, 0 otherwise
     - a 0/1 value: 1 if P1 is a prefix of P2, 0 otherwise
     - a scalar: if P1 is a prefix of P2, the number of lines that need to be added to P1 to get P2; negative otherwise
     # - a 0/1 value: 1 if P1 is a subset of P2, negative otherwise
     # - a scalar: the number of lines needed to turn P1 into P2 if P1 is a subset of P2; negative otherwise
     # - a scalar measuring the likelihood of bitmap sets B1 and B2 being generated by the same function
    """
    if dir_name is None:
        code = make_name_code(n_exs=n_exs, scene_sizes=scene_sizes, shape_types=shape_types, n_zs=n_zs, n_envs=n_envs)
        dir_name = f'../data/{code}-discrim-combi'

    min_shapes, max_shapes = scene_sizes
    n_sizes = max_shapes - min_shapes + 1
    assert n_exs % (2 * n_sizes) == 0, \
        f'number of training examples should be divisible by the number of shapes in each scene times 2'
    
    # Make envs and arithmetic exprs
    envs1 = [{'z': seed_zs(), 'sprites': seed_sprites()} for _ in range(n_envs)]
    envs2 = [{'z': seed_zs(), 'sprites': seed_sprites()} for _ in range(n_envs)]
    a_exprs = [a_expr for a_expr, size in bottom_up_generator(a_depth, a_grammar, envs1 + envs2)]
    shapes = make_shapes(shape_types=shape_types, envs=envs1+envs2, a_exprs=a_exprs, n_zs=n_zs, scene_sizes=scene_sizes,
                         save_dir=dir_name, n_shapes=n_exs * max_shapes, enum=enum_all_shapes,
                         n_processes=n_processes)

    # Generate pos/neg examples
    # - pos: use two different env sets on the same program
    # - neg: use the same env set on two different programs
    
    fname = f'{dir_name}/discrim-combi.exs'
    shapes = list(shapes)
    scenes = []  # store lists of objs
    for size in range(min_shapes, max_shapes+1):
        for i in range(n_exs//(2 * n_sizes)):
            scenes.append(gen_random_scene(shapes, envs1 + envs2, size))
    shuffle(scenes)
    
    # positive examples
    cleared = False
    for scene in scenes:
        prog = Seq(*scene)
        bmps1, bmps2 = prog.eval(envs1), prog.eval(envs2)
        util.save(((bmps1, bmps2), [1, 1, 0]), fname, append=cleared, verbose=not cleared)
        cleared = True
    # negative examples
    for scene in scenes[:len(scenes)//2]:
        prog1 = Seq(*scene)
        for i in range(1, len(scene)):
            # prefix program
            prog2 = Seq(*scene[:i])
            bmps1, bmps2 = prog1.eval(envs1), prog2.eval(envs2)
            util.save(((bmps1, bmps2), [0, 1, len(scene) - i]), fname, append=cleared, verbose=not cleared)
            
            # completely different program
            scene2 = choice(scenes)
            if scene2 == scene or all(obj in scene for obj in scene2): continue
            prog2 = Seq(*scene2)
            bmps1, bmps2 = prog1.eval(envs1), prog2.eval(envs2)
            util.save(((bmps1, bmps2), [0, 0, -1]), fname, append=cleared, verbose=not cleared)

def get_lines(seq):
    try:
        return seq.bmps
    except AttributeError:
        return []

def eval_expr(expr, env):
    try:
        return expr.eval(env)
    except (AssertionError, AttributeError):
        return T.zeros(B_H, B_W)

def make_discrim_ex(e_expr, o_expr, envs):
    assert isinstance(e_expr, Seq), "Found an input expression that isn't a Seq"

    equal = e_expr == o_expr
    e_lines = get_lines(e_expr)
    o_lines = get_lines(o_expr)
    prefix = util.is_prefix(o_lines, e_lines)
    dist = len(e_lines) - len(o_lines)
    
    e_bmps = T.stack([eval_expr(e_expr, env) for env in envs])
    o_bmps = T.stack([eval_expr(o_expr, env) for env in envs])
    bmps = T.cat((e_bmps, o_bmps)).unsqueeze(0)
    
    example = (bmps, [equal, prefix, dist], e_expr, o_expr)
    return example

def make_discrim_exs_model_perturb(shapes_loc, model_checkpoint_loc, data_glob, dir_name,
                                   N, d_model, batch_size, max_p_len=50, n_processes=1, verbose=False):
    """Read in reference examples and perturb them by running the model"""
    # Load in trained model, data
    model = recover_model(checkpoint_loc=model_checkpoint_loc,
                          name=f'{dir_name}-model',
                          N=N, H=B_H, W=B_W, d_model=d_model, batch_size=batch_size)
    dataloader = model.make_dataloader(lambda: util.load_multi_incremental(data_glob))

    # Read in environments from shapes file
    envs = util.load(shapes_loc)['envs']
    assert len(envs) == model.N
    
    # Generate examples by using inference on model
    fname = f'{dir_name}.model-perturbed.discrim.exs'
    cleared = False
    for B, P in dataloader:
        d = model.sample_programs(B, P, max_length=max_p_len)
        e_exprs, o_exprs = d['expected exprs'], d['out exprs']

        with mp.Pool(n_processes) as pool:
            exs = pool.starmap(make_discrim_ex,
                               ((e_expr, o_expr, envs) for e_expr, o_expr in zip(e_exprs, o_exprs)))
        for ex in exs:
            if verbose and ex[-1] != ex[-2]:
                print(f"Generated example: {ex[-2]} -> {ex[-1]}")
            util.save(ex, fname, append=cleared, verbose=not cleared)
            cleared = True

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

def test_shape_blankness(shapes_loc):
    log = util.load(shapes_loc)
    shapes = log['shapes']
    envs = log['envs']

    print(f"envs: {[env['z'] for env in envs]}")
    for shape in shapes:  # Rect(Num(0), Z(2), Num(4), Num(5))
        print(shape)
        # if shape.zs(): viz.viz_mult([eval_expr(shape, env) for env in envs], text=f'{shape}')
        for env in envs:
            assert not (eval_expr(shape, env) == 0).all(), \
                f"Found blank shape: {shape} w/ zs {env['z']}"
    
    print(" [+] passed test_shape_blankness")

def test_expr_shape_blankness(exs_loc, envs_loc):
    envs = util.load(envs_loc)['envs']
    exs = util.load_multi_incremental(exs_loc)
    
    for B, P in exs:
        print(P)

def make_name_code(n_exs, scene_sizes, shape_types, n_zs, n_envs):
    
    def shape_code(shape_types):
        return f"{'p' if Point in shape_types else ''}" \
               f"{'l' if Line in shape_types else ''}" \
               f"{'r' if Rect in shape_types else ''}"
    
    def n_code(sizes):
        a, b = min(sizes), max(sizes)
        if a == b:
            return str(a)
        else:
            return f'{a}~{b}'
    
    sz_code = n_code(scene_sizes)
    sh_code = shape_code(shape_types)
    z_code = n_code(n_zs)
    code = f"{n_exs}-{sz_code}{sh_code}{z_code}z{n_envs}e"
    return code


if __name__ == '__main__':
    # test_rm_dead_code()
    # test_canonical_ordering()
    g = Grammar(ops=[Plus, Minus, Times],
                consts=([Z(i) for i in range(LIB_SIZE)] + [Num(i) for i in range(0, 10)]))
    print(g.ops, g.consts)
    # for mode in ['train', 'test']:
    #     make_tf_exs(
    #         dir_name=f'../data/100-r0~1z5e-nonblank/{mode}',
    #         n_exs=100,
    #         shape_types=[Rect],
    #         scene_sizes=(1, 5),
    #         n_envs=5,
    #         n_zs=(0, 1),
    #         enum_all_shapes=False,
    #         label_zs=True,
    #         a_grammar=g,
    #         a_depth=1,
    #         n_processes=8,
    #     )
    # make_discrim_exs_combi(
    #     n_exs=10,
    #     shape_types=[Rect],
    #     scene_sizes=(1, 5),
    #     n_envs=1,
    #     n_zs=(0, 0),
    #     a_grammar=g,
    #     a_depth=1,
    #     enum_all_shapes=False,
    #     n_processes=16
    # )
    for mode in ['train', 'test']:
        make_discrim_exs_model_perturb(
            model_checkpoint_loc='/home/djl328/arc/models/256m-5lr/tf_model_256m-5lr_33.pt',
            shapes_loc='/home/djl328/arc/data/1mil/r0~1z5e/*test.tf.cmps',
            data_glob='/home/djl328/arc/data/1mil/r0~1z5e/*test*.tf.exs',
            dir_name=f'/home/djl328/arc/data/1mil/perturbed-r0~1z5e/{mode}',
            N=5, d_model=256, batch_size=32, max_p_len=50,
            n_processes=16,
            verbose=True
        )
