"""
Make large programs as training data
"""

import pickle
from math import floor, ceil, log2
import random as R
import numpy as np

from viz import viz, viz_grid, viz_mult
from grammar import *
from bottom_up import bottom_up_generator, eval

def gen_shape_pool(entities, aexprs, envs, n):
    zexprs, cexprs = util.split(aexprs, lambda a: a.zs())
    pool = {}
    for entity in entities:
        n_misses = 0
        pool[entity] = []
        while len(pool[entity]) < n:
            color = rand_color()
            k = R.randrange(1, len(entity.in_types)) # num z's to use
            args = ([R.choice(zexprs) for _ in entity.in_types[:k]] + 
                    [R.choice(cexprs) for _ in entity.in_types[k:]])
            R.shuffle(args)
            e = entity(*args[:-1], color)
            outs = eval(e, envs)
            if all(out is not None for out in outs): 
                pool[entity].append(e)
            else:
                n_misses += 1    
            print(f'{entity} misses:', n_misses, end='\r')
        print()
    return pool

def gen_random_expr(pool, aexprs, envs, n_objs):
    objs = []
    for _ in range(n_objs):
        n = R.randint(0, 10)
        if n < 8:
            if n < 1:   entity = Point
            elif n < 4: entity = Line
            else:       entity = Rect
            e = R.choice(pool[entity])
        else:
            i = R.randrange(1, LIB_SIZE)
            e = rand_sprite(i, rand_color(), envs, aexprs)
        # TODO: transforms
        objs.append(e)

    print()
    return Seq(*objs)

def gen_random_exprs(pool, aexprs, envs, n_exprs, n_objs):
    for i in range(n_exprs):
        expr = gen_random_expr(pool, aexprs, envs, n_objs)
        print(f'expr generated [{i+1}/{n_exprs}]: {expr}')
        yield expr

def rand_color():
    return Num(R.randint(1, 9))

def rand_sprite(i, color, envs, aexprs):
    n_misses = 0
    while True:
        s = Sprite(i, R.choice(aexprs), R.choice(aexprs))
        outs = eval(s, envs)
        if all(out is not None for out in outs):
            break
        else: 
            n_misses += 1
            viz_mult([s.eval(env) for env in envs[:10]])
        print(f'sprite misses:', n_misses, end='\r')
    print()
    if color > 1: return Apply(Recolor(color), s)
    else:         return s

def rand_transform(e):
    '''
    Return a random (nontrivial) transformation of e
    '''
    # # Transform
    # n = R.choice(aexprs)
    # t = R.choice(transforms)
    # t_args = [R.choice(aexprs) for _ in t.in_types]
    # if any(n.eval(env) > 0 for env in envs):
    #     t = Repeat(t(*t_args), n)
    # else:
    #     t = t(*t_args)
    pass

def simplify(sprites, env):
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

def save_expr(expr):
    print(f'Saving expr {expr}...')
    with open('../data/exprs.dat', 'ab') as f:
        pickle.dump(expr, f)

def save_envs(envs):
    print('Saving zs...')
    with open('../data/zs.dat', 'wb') as f:
        pickle.dump(zs, f)

def save_viz(expr, env, i, j):
    try:
        viz(expr.eval(env), 
            f'e{i}-{j}', str(expr), 
            fname=f'../data/viz/exprs-2/e{i}-{j}.png', 
            show=False)
    except AssertionError:
        pass

if __name__ == '__main__':
    n_exprs = 1000
    n_envs = 100
    abound = 1
    n_objs = 4
    print(f'Parameters used: n_exprs={n_exprs}, n_envs={n_envs}, a_bound={abound}, n_objs={n_objs}')

    ag = Grammar(ops=[Plus, Minus, Times], 
                 consts=([Z(i) for i in range(LIB_SIZE)] + 
                         [Num(i) for i in range(Z_LO, Z_HI + 1)]))
    envs = [{'z': seed_zs(), 'sprites':seed_sprites()} for _ in range(n_envs)]
    aexprs = [e for e, s in bottom_up_generator(abound, ag, envs)]
    pool = gen_shape_pool([Point, Line, Rect], aexprs, envs, 10)
    gen = gen_random_exprs(pool, aexprs, envs, n_exprs, n_objs)

    # save_envs(envs)
    # open('../data/exprs.dat', 'w').close() # clear prior exprs.dat file contents

    k = 5
    for i, expr in enumerate(gen):
        viz_grid([expr.eval(env) for env in envs[:k**2]], k, txt=str(expr))
        # save_expr(expr)

        # save_viz first 50 f, each with first 10 z's
        # if i < 50:
        #     for j, env in enumerate(envs[:10]):
        #         save_viz(expr, env, i, j)

