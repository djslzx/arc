"""
Make large programs as training data
"""

import pickle
from math import floor, ceil, log2
import random as R
import numpy as np

from viz import viz, viz_grid
from grammar import *
from bottom_up import bottom_up_generator, eval

def gen_zs(n):
    zs = (T.rand(n, LIB_SIZE) * (Z_HI - Z_LO) - Z_LO).long()
    return zs

def split(l, f):
    sat, unsat = [], []
    for x in l:
        if f(x): sat.append(x)
        else:    unsat.append(x)
    return sat, unsat

def gen_random_expr(zs, n_objs, aexprs):
    envs = [{'z': z} for z in zs]
    objects = [Line, Rect]
    transforms = [Translate]
    colors = list(range(1, 10))
    zexprs, cexprs = split(aexprs, lambda a: a.zs())

    objs = []
    n_misses = 0
    while len(objs) < n_objs:
        # # Transform
        # n = R.choice(aexprs)
        # t = R.choice(transforms)
        # t_args = [R.choice(aexprs) for _ in t.in_types]
        # if any(n.eval(env) > 0 for env in envs):
        #     t = Repeat(t(*t_args), n)
        # else:
        #     t = t(*t_args)

        # Entity
        f = R.choice(objects)
        k = R.randint(1, len(f.in_types) - 1)
        # k = 0 if f == Shape else min(np.random.geometric(0.5), len(f.in_types) - 1) 
        f_args = [R.choice(zexprs) for _ in f.in_types[:k]] + [R.choice(cexprs) for _ in f.in_types[k:]]
        R.shuffle(f_args)
        *f_args, color = f_args

        # e = Apply(t, f(*f_args, color=color))
        e = f(*f_args, color=color)

        outs = eval(e, envs)
        if all(out is not None for out in outs) and all(color.eval(env) > 0 for env in envs):
            objs.append(e)
        else: 
            n_misses += 1    
        print("Misses:", n_misses, end='\r')

    print()
    return Seq(*objs)

def rand_sprite(i, env):
    def gen():
        x = random.randint(0, B_W-1)
        y = random.randint(0, B_H-1)
        color = random.randint(1, 9)
        s = Sprite(i, Num(x), Num(y))
        if color > 1: return Apply(Recolor(Num(color)), s)
        else:         return s
        
    while ant.classify((sprite := gen()).eval(env)) != 'Sprite': pass
    return sprite

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

def gen_random_exprs(zs, n_exprs, n_objs, aexprs):
    for i in range(n_exprs):
        expr = gen_random_expr(zs, n_objs, aexprs)
        print(f'expr generated [{i+1}/{n_exprs}]: {expr}')
        yield expr

def save_expr(expr):
    print(f'Saving expr {expr}...')
    with open('../data/exprs.dat', 'ab') as f:
        pickle.dump(expr, f)

def save_zs(zs):
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
    n_zs = 100
    abound = 1
    n_objs = 4
    print(f'Parameters used: n_exprs={n_exprs}, n_zs={n_zs}, a_bound={abound}, n_objs={n_objs}')

    zs = gen_zs(n_zs)
    g = Grammar(ops=[Plus, Minus, Times], 
                consts=([Z(i) for i in range(LIB_SIZE)] + 
                        [Num(i) for i in range(Z_LO, Z_HI + 1)]))
    envs = [{'z':z} for z in zs]
    aexprs = [e for e, s in bottom_up_generator(abound, g, envs)]
    # for a in aexprs: print(a)

    # print(nest_stacks([Num(i) for i in range(10)]))
    # expr = gen_random_expr(zs, n_objs, aexprs, 0.5)

    # save_zs(zs)
    # open('../data/exprs.dat', 'w').close() # clear prior exprs.dat file contents

    gen = gen_random_exprs(zs, n_exprs, n_objs, aexprs)
    k = 5
    for i, expr in enumerate(gen):
        viz_grid([expr.eval(env) for env in envs[:k**2]], k, txt=str(expr))
        # save_expr(expr)

        # save_viz first 50 f, each with first 10 z's
        # if i < 50:
        #     for j, env in enumerate(envs[:10]):
        #         save_viz(expr, env, i, j)

