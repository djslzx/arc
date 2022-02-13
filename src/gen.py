"""
Make programs to use as training data
"""

import pdb
import pickle
from math import floor, ceil, log2, sqrt
import random as R
import numpy as np

from viz import viz, viz_grid, viz_mult
from grammar import *
from bottom_up import bottom_up_generator, eval


def gen_shape_pool(entities, a_exprs, envs, pool_size):
    z_exprs, c_exprs = util.split(a_exprs, lambda a: a.zs())
    pool = {}
    outs = {}
    for entity in entities:
        n_tries = 0
        n_hits = 0
        pool[entity] = []
        while len(pool[entity]) < pool_size:
            color = rand_color()
            k = R.randrange(1, len(entity.in_types)) # num z's to use
            args = ([R.choice(z_exprs) for _ in entity.in_types[:k]] + 
                    [R.choice(c_exprs) for _ in entity.in_types[k:]])
            R.shuffle(args)
            e = entity(*args[:-1], color)
            outs = eval(e, envs)
            if all(out is not None for out in outs): 
                n_hits += 1
                pool[entity].append(e)
            n_tries += 1    
        print(f'{entity} hits: {n_hits}/{n_tries}')
    return pool

def rm_dead_code(entities, envs):
    '''
    Remove any occluded entities
    '''
    def bmps(xs):
        return T.stack([Seq(*xs).eval(env) for env in envs])

    orig_bmps = bmps(entities)
    keep = []
    for i, entity in enumerate(entities):
        # pdb.set_trace()
        excl_bmps = bmps(keep + entities[i+1:])
        if (excl_bmps != orig_bmps).any():
            keep.append(entity)
    return keep

def canonical_ordering(entities):
    '''
    Order entities by complexity and position.
    - Complexity: Point < Line < Rect 
    - Positions are sorted in y-major order (e.g., (1, 4) < (2, 1) and (3, 1) < (3, 2))
    '''
    def destructure(e):
        if isinstance(e, Point):
            return (0, e.x, e.y)
        elif isinstance(e, Line):
            return (1, e.x1, e.y1, e.x2, e.y2)
        elif isinstance(e, Rect):
            return (2, e.x, e.y, e.w, e.h)
    
    return sorted(entities, key=lambda e: destructure(e))

def gen_random_expr(pool, a_exprs, envs, n_entities):
    '''
    Generate a random sequence of entities.  
    - dead code removal
    - canonical ordering
    '''
    # TODO: transforms

    entities = []
    while len(objs) < n_entities:
        n = R.randint(0, 8)     # 0..8 (inclusive)
        if n < 1:   t = Point
        elif n < 5: t = Line
        else:       t = Rect
        entity = R.choice(pool[t])
        if entity not in objs:
            objs.append(entity)
        entities = rm_dead_code(entities, envs)
        
    entities = canonical_ordering(entities)
    expr = Seq(*entities)
    return clean(expr)

def gen_random_exprs(pool, a_exprs, envs, n_exprs, n_entities, verbose=True):
    for i in range(n_exprs):
        expr = gen_random_expr(pool, a_exprs, envs, n_entities)
        if verbose: print(f'expr generated [{i+1}/{n_exprs}]: {expr}')
        yield expr

def rand_color():
    return Num(R.randrange(1, 10))

def rand_sprite(envs, a_exprs, i=-1, color=-1):
    i = i if 0 <= i < LIB_SIZE else R.randrange(1, LIB_SIZE)
    color = color if isinstance(color, Num) and 1 <= color.n <= 9 else rand_color()
    n_misses = 0
    while True:
        s = Sprite(i, R.choice(a_exprs), R.choice(a_exprs))
        outs = eval(s, envs)
        if all(out is not None for out in outs):
            break
        else: 
            n_misses += 1
    print(f'sprite misses:', n_misses)
    if color > 1: return Apply(Recolor(color), s)
    else:         return s

def rand_transform(e):
    '''
    Return a random (nontrivial) transformation of e
    '''
    # # Transform
    # n = R.choice(a_exprs)
    # t = R.choice(transforms)
    # t_args = [R.choice(a_exprs) for _ in t.in_types]
    # if any(n.eval(env) > 0 for env in envs):
    #     t = Repeat(t(*t_args), n)
    # else:
    #     t = t(*t_args)
    pass

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
        [0, 1,],
        [0, 2,],
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

def make_exprs(n_exprs,         # number of total programs to make
               n_envs,          # number of envs (bitmaps) per program
               max_n_entities,  # number of entities in each program
               a_grammar,       # grammar for arithmetic exprs (components of entities)
               a_bound,         # bound on arithmetic exprs in each entity
               entities,        # entity classes allowed
               cmps_loc,        # where to save/load pool of random components
               exprs_loc,       # where to save generated programs
               load_pool=True): # whether to load cmps from cmps_loc or not (gen from scratch)
    n_exprs_per_size = n_exprs//max_n_entities
    pool_size = n_exprs * max_n_entities
    print(f'Parameters: n_exprs={n_exprs}, n_envs={n_envs}, max_n_entities={max_n_entities}, a_bound={a_bound}, entities={entities}')

    if load_pool:
        cmps = util.load(cmps_loc)
        envs, pool, a_exprs = cmps['envs'], cmps['pool'], cmps['a_exprs']
        meta = cmps['meta']
        n_envs, a_bound, pool_size, entities = meta['n_envs'], meta['a_bound'], meta['pool_size'], meta['entities']
    else:
        envs = [{'z': seed_zs(), 'sprites': seed_sprites()} for _ in range(n_envs)]
        a_exprs = [a_expr for a_expr, size in bottom_up_generator(a_bound, a_grammar, envs)]
        pool = gen_shape_pool(entities, a_exprs, envs, pool_size)
        try:
            open(cmps_loc, 'wb').close() # clear file
        except FileNotFoundError:
            pass
        util.save({'meta': {'n_envs': n_envs,
                            'a_bound': a_bound,
                            'pool_size': pool_size,
                            'entities': entities}, 
                   'envs': envs, 
                   'pool': pool, 
                   'a_exprs': a_exprs},
                  cmps_loc)

    # Generate and save exprs w/ bmp outputs
    for n_entities in range(1, max_n_entities+1):
        for expr in gen_random_exprs(pool, a_exprs, envs, n_exprs_per_size, n_entities):
            bmps = [expr.eval(env) for env in envs] 
            p = expr.simplify_indices().serialize()
            util.save((bmps, p), fname=exprs_loc, append=True, verbose=False)

def viz_exs(fname):
    for bmps, tokens in util.load_incremental(fname):
        print('tokens:', tokens)
        expr = deserialize(tokens)
        viz_grid(bmps[:9], expr)

def list_exs(fname):
    for bmps, tokens in util.load_incremental(fname):
        print(deserialize(tokens))

if __name__ == '__main__':

    test_rm_dead_code()
    test_canonical_ordering()

    # # Load saved exprs and generate bmps
    # data = util.load('../data/exs.dat')
    # for bmps, tokens in data:
    #     print('tokens:', tokens)
    #     d = deserialize(tokens)
    #     print('expr:', d, len(d))
    #     print(viz_grid(bmps[:25], d))

    # make_exprs(n_exprs=5, n_envs=9, max_n_entities=5, a_bound=1,
    #            entities=[Point, Line, Rect],
    #            a_grammar = Grammar(ops=[Plus, Minus, Times], 
    #                                consts=([Z(i) for i in range(LIB_SIZE)] + 
    #                                        [Num(i) for i in range(Z_LO, Z_HI + 1)]))
    #            cmps_loc='../data/tiny-cmps.dat',
    #            exprs_loc='../data/tiny-exs.dat',
    #            load_pool=False)

    # list_exs('../data/full-exs.dat')
