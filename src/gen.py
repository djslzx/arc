"""
Make large programs as training data
"""

import pickle
from math import floor, ceil, log2, sqrt
import random as R
import numpy as np

from viz import viz, viz_grid, viz_mult
from grammar import *
from bottom_up import bottom_up_generator, eval


def gen_shape_pool(entities, a_exprs, envs, n):
    z_exprs, c_exprs = util.split(a_exprs, lambda a: a.zs())
    pool = {}
    for entity in entities:
        n_tries = 0
        n_hits = 0
        pool[entity] = []
        while len(pool[entity]) < n:
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
            print(f'{entity} hits: {n_hits}/{n_tries}', end='\r')
        print()
    return pool

def gen_random_expr(pool, a_exprs, envs, n_objs):
    objs = []
    for _ in range(n_objs):
        n = R.randint(0, 10)
        if n < 8:
            if n < 1:   entity = Point
            elif n < 4: entity = Line
            else:       entity = Rect
            e = R.choice(pool[entity])
        else:
            e = rand_sprite(envs, a_exprs)
        # TODO: transforms
        objs.append(e)
    return Seq(*objs)

def gen_random_exprs(pool, a_exprs, envs, n_exprs, n_objs):
    for i in range(n_exprs):
        expr = gen_random_expr(pool, a_exprs, envs, n_objs)
        print(f'expr generated [{i+1}/{n_exprs}]: {expr}')
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
        print(f'sprite misses:', n_misses, end='\r')
    if n_misses > 0: print()
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

def simplify(sprites, env):
    '''
    Remove any sprites in `sprites` that are occluded by other sprites in `env`
    '''
    keep = sprites[:1]
    for i, sprite in enumerate(sprites[1:], 1):
        # check that the sprites overlayed on top of `sprite` don't obscure `sprite`
        e = Seq(*sprites[:i])
        e_bmp = e.eval(env)
        s_bmp = sprite.eval(env)
        if not all(s_bmp[r][c] == 0 or e_bmp[r][c] > 0 
                   for r in range(e_bmp.shape[0]) 
                   for c in range(e_bmp.shape[1])):
            keep.append(sprite)
    return keep

def viz_sprites(envs):
    k = floor(sqrt(LIB_SIZE))
    for env in envs:
        viz_grid(env['sprites'][:k**2], k, txt=f'sprites[:{k**2}]')

def test_simplify():
    tests = [
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
    ]

    for lib, sprites, ans in tests:
        lib = [util.img_to_tensor(b, w=B_W, h=B_H) for b in lib]
        out = simplify(sprites, {'z': [], 'sprites': lib})
        assert out == ans, f"Failed test: expected {ans}, got {out}"
    print("[+] passed test_simplify")


if __name__ == '__main__':
    n_exprs = 100
    n_envs = 100
    a_bound = 1
    n_objs = 4
    pool_size = 100
    entities = [Point, Line, Rect]
    print(f'Parameters used: n_exprs={n_exprs}, n_envs={n_envs}, a_bound={a_bound}, n_objs={n_objs}')

    ag = Grammar(ops=[Plus, Minus, Times], 
                 consts=([Z(i) for i in range(LIB_SIZE)] + 
                         [Num(i) for i in range(Z_LO, Z_HI + 1)]))

    # Generate and save pool
    # envs = [{'z': seed_zs(), 
    #          'sprites':seed_sprites()} 
    #         for _ in range(n_envs)]
    # a_exprs = [expr for expr, size in bottom_up_generator(a_bound, ag, envs)]
    # pool = gen_shape_pool(entities, a_exprs, envs, pool_size)
    # util.save({'meta': {'n_envs': n_envs,
    #                     'a_bound': a_bound,
    #                     'pool_size': pool_size,
    #                     'entities': entities}, 
    #            'envs': envs, 
    #            'pool': pool, 
    #            'a_exprs': a_exprs},
    #           '../data/cmps.dat')

    # Load saved pool
    # cmps = util.load('../data/cmps.dat')
    # envs, pool, a_exprs = cmps['envs'], cmps['pool'], cmps['a_exprs']
    # meta = cmps['meta']
    # n_envs, a_bound, pool_size, entities = meta['n_envs'], meta['a_bound'], meta['pool_size'], meta['entities']

    # Generate and save exprs w/ bmp outputs
    # exprs = gen_random_exprs(pool, a_exprs, envs, n_exprs, n_objs)
    # data = []
    # for i, expr in enumerate(exprs):
    #     serialized = expr.serialize()
    #     bmps = [expr.eval(env) for env in envs] 
    #     print(' ', serialized, len(bmps))
    #     data.append((bmps, serialized))
    # util.save(data, '../data/exs.dat')

    # Load saved exprs and generate bmps
    data = util.load('../data/exs.dat')
    for tokens, bmps in data:
        print('tokens:', tokens)
        d = deserialize(tokens)
        print('expr:', d, len(d))
        print(viz_grid(bmps[:25], 5, d))
