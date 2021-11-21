"""
Make deep programs as training data for perceptual loss.
"""

from math import floor, ceil, log2
from random import choice 
from grammar_visitor import *
from bottom_up import bottom_up_generator, eval
from viz import viz

def gen_zs(n):
    zs = (T.rand(n, Z_SIZE) * (Z_HI - Z_LO) - Z_LO).long()
    return zs

def nest_stacks(elts):
    if len(elts) == 1:
        return elts[0]
    else:
        return Stack(nest_stacks(elts[:1]), 
                     nest_stacks(elts[1:]))

def gen_random_expr(zs, n_objs, aexprs, k):
    envs = [{'z': z} for z in zs]
    objects = [Point, Line, Rect]
    colors = list(range(1, 10))

    objs = []
    n_misses = 0
    while len(objs) < n_objs:
        f = choice(objects)
        color = choice(colors)
        args = [choice(aexprs) for t in f.in_types]
        e = f(*args, color)
        
        # check that e is well-formed
        outs = eval(e, envs)
        ok = sum(out is not None for out in outs)
        if ok >= k * len(zs):
            objs.append(e)
        else:
            n_misses += 1    
        print("Misses:", n_misses, end='\r')

    print()
    return nest_stacks(objs)

def gen_random_exprs(zs, n_exprs, n_objs, aexprs, k):
    for i in range(n_exprs):
        expr = gen_random_expr(zs, n_objs, aexprs, k)
        print(f'expr generated [{i+1}/{n_exprs}]: {expr}')
        yield expr

def save(zs, exprs):
    print('Saving zs and exprs...')
    with open('../data/zs.dat', 'wb') as f:
        pickle.dump(zs, f)
    with open('../data/exprs.dat', 'wb') as f:
        for expr in exprs:
            pickle.dump(expr, f)

def visualize(expr, env, i, j):
    try:
        viz(expr.eval(env), f'e{i}-{j}', str(expr), 
            f'../data/expr-viz/e{i}-{j}.png', 
            show=False, save=True)
    except AssertionError:
        pass

if __name__ == '__main__':
    n_exprs = 1000
    n_zs = 1000
    abound = 1
    n_objs = 3

    zs = gen_zs(n_zs)
    g = Grammar(ops=[Plus, Minus, Times], 
                consts=([Z(i) for i in range(Z_SIZE)] + 
                        [Num(i) for i in range(Z_LO, Z_HI + 1)]))
    envs = [{'z':z} for z in zs]
    aexprs = [e for e, s in bottom_up_generator(abound, g, envs)]
    # for a in aexprs: print(a)

    # print(nest_stacks([Num(i) for i in range(10)]))
    # expr = gen_random_expr(zs, n_objs, aexprs, 0.5)

    gen = gen_random_exprs(zs, n_exprs, n_objs, aexprs, 1.0)
    exprs = []
    for i, expr in enumerate(gen):
        exprs.append(expr)

        # visualize first 50 f, each with first 10 z's
        if i < 50:
            for j, env in enumerate(envs[:10]):
                visualize(expr, env, i, j)

    # save(zs, exprs)

