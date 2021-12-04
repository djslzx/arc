"""
Make deep programs as training data for perceptual loss.
"""

import pickle
from math import floor, ceil, log2
from random import choice 

from viz import viz
from grammar_visitor import *
from bottom_up import bottom_up_generator, eval

def gen_zs(n):
    zs = (T.rand(n, Z_SIZE) * (Z_HI - Z_LO) - Z_LO).long()
    return zs

def nest_stacks(elts):
    if len(elts) == 1:
        return elts[0]
    else:
        return Stack(nest_stacks(elts[:1]), 
                     nest_stacks(elts[1:]))

def gen_random_expr(zs, n_objs, aexprs):
    envs = [{'z': z} for z in zs]
    objects = [Line, Rect]
    colors = list(range(1, 10))

    objs = []
    n_misses = 0
    while len(objs) < n_objs:
        f = choice(objects)
        color = choice(colors)
        args = [choice(aexprs) for t in f.in_types]
        e = f(*args, color)
        
        # check that e is well-formed and includes at least one random component (avoid constant imgs)
        outs = eval(e, envs)
        ok = len(e.zs()) > 0 and all(out is not None for out in outs)
        if ok: objs.append(e)
        else: n_misses += 1    
        print("Misses:", n_misses, end='\r')

    print()
    return nest_stacks(objs)

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
    n_objs = 3
    print(f'Parameters used: n_exprs={n_exprs}, n_zs={n_zs}, a_bound={abound}, n_objs={n_objs}')

    zs = gen_zs(n_zs)
    g = Grammar(ops=[Plus, Minus, Times], 
                consts=([Z(i) for i in range(Z_SIZE)] + 
                        [Num(i) for i in range(Z_LO, Z_HI + 1)]))
    envs = [{'z':z} for z in zs]
    aexprs = [e for e, s in bottom_up_generator(abound, g, envs)]
    # for a in aexprs: print(a)

    # print(nest_stacks([Num(i) for i in range(10)]))
    # expr = gen_random_expr(zs, n_objs, aexprs, 0.5)

    # save_zs(zs)
    # open('../data/exprs.dat', 'w').close() # clear prior exprs.dat file contents

    gen = gen_random_exprs(zs, n_exprs, n_objs, aexprs)
    for i, expr in enumerate(gen):
        for env in envs[:10]:
            viz(expr.eval(env))
        # save_expr(expr)

        # save_viz first 50 f, each with first 10 z's
        # if i < 50:
        #     for j, env in enumerate(envs[:10]):
        #         save_viz(expr, env, i, j)

