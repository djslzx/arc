"""
Make programs to use as training data
"""
import pdb
import math
import time
import multiprocessing as mp

from src.grammar import *
from src.bottom_up import bottom_up_generator, eval
from src.transformer import recover_model

def gen_random_shape(shape_type, c_exprs: list[Expr], z_exprs: list[Expr], z_range: tuple[int, int]):
    in_types = shape_type.in_types
    # NOTE: this exploits the fact that all shapes have arguments of the same type (Nums).
    # If this doesn't hold, the below code won't work.
    assert all(t == 'int' for t in in_types)
    n_zs = min(len(in_types), random.randint(*z_range))  # cap n_zs by number of args for shape type
    z_args = [random.choice(z_exprs) for _ in range(n_zs)]
    c_args = [random.choice(c_exprs) for _ in range(len(in_types) - n_zs)]
    args = z_args + c_args
    random.shuffle(args)
    return shape_type(*args)

def gen_shapes(n_shapes, a_exprs, envs, shape_types, min_zs, max_zs, verbose=True):
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
            shape = gen_random_shape(shape_type, c_exprs, z_exprs, (min_zs, max_zs))
            if shape not in shapes and all(out is not None for out in eval(shape, envs)):
                shapes.add(shape)
                n_hits += 1
                if verbose and n_hits % 1000 == 0:
                    print(f'{shape_type} hits: {n_hits}/{n_tries}')
            if verbose and n_tries % 10000 == 0:
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
        shape = random.choice(shapes)
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
    i = i if 0 <= i < LIB_SIZE else random.randrange(1, LIB_SIZE)
    color = color if isinstance(color, Num) and 1 <= color.n <= Z_HI else Num(random.randint(1, Z_HI))
    n_misses = 0
    while True:
        s = Sprite(i, random.choice(a_exprs), random.choice(a_exprs))
        outs = eval(s, envs)
        if all(out is not None for out in outs):
            break
        else:
            n_misses += 1
    print(f'sprite misses:', n_misses)
    return Apply(Recolor(color), s) if color > 1 else s

def rand_transform():
    """
    Return a random (nontrivial) transformation of e
    """
    # # Transform
    # n = random.choice(a_exprs)
    # t = random.choice(transforms)
    # t_args = [random.choice(a_exprs) for _ in t.in_types]
    # if any(n.eval(env) > 0 for env in envs):
    #     t = Repeat(t(*t_args), n)
    # else:
    #     t = t(*t_args)
    assert False, "not implemented"

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

def eval_expr(expr, env):
    try:
        return expr.eval(env)
    except (AssertionError, AttributeError):
        return T.zeros(B_H, B_W)

def enum_envs():
    for z in it.product(*[range(Z_LO, Z_HI+1) for _ in range(LIB_SIZE)]):
        yield {'z': z}

def compute_likelihood(observed_bitmaps, fitted_program):
    """
    Compute log P(B | p) = sum{b in B} log sum{z in Z} I[p(z) = b] - |B| log |Z|, where
      - `p`: working program (fitted_program)
      - `Z`: set of all sources of randomness (envs)
      - `B`: bitmaps observed by model
    """
    sampled_bitmaps = [fitted_program(env) for env in enum_envs()]
    n_envs = ((Z_HI - Z_LO + 1) ** LIB_SIZE)
    n_bitmaps = len(observed_bitmaps)
    score = 0
    for observed_bitmap in observed_bitmaps:
        n_matches = 0
        for sampled_bitmap in sampled_bitmaps:
            n_matches += T.equal(sampled_bitmap, observed_bitmap)
        score += math.log(n_matches)
    return score - n_bitmaps * math.log(n_envs)

def make_discrim_ex(source_program, fitted_program, envs):
    assert isinstance(source_program, Seq), "Found an input expression that isn't a Seq"

    equal = source_program == fitted_program
    lines_true = source_program.lines()
    lines_fitted = fitted_program.lines()
    prefix = util.is_prefix(lines_fitted, lines_true)
    dist = len(lines_true) - len(lines_fitted)
    
    observed_bitmaps = T.stack([eval_expr(source_program, env) for env in envs])
    sampled_bitmaps = T.stack([eval_expr(fitted_program, env) for env in envs])
    bmps = T.cat((observed_bitmaps, sampled_bitmaps)).unsqueeze(0)

    likelihood = compute_likelihood(observed_bitmaps, fitted_program)
    
    example = (bmps, [equal, prefix, dist, likelihood], source_program, fitted_program)
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
        source_programs, fitted_programs = d['expected exprs'], d['out exprs']

        for source_program, fitted_program in zip(source_programs, fitted_programs):
            ex = make_discrim_ex(source_program, fitted_program, envs)
            if verbose and ex[-1] != ex[-2]:
                print(f"Generated example: {ex[-2]} -> {ex[-1]}")
            util.save(ex, fname, append=cleared, verbose=not cleared)
            cleared = True

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
