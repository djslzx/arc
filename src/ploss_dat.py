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

def nest_stacks(elts, n):
    if n == 1:
        return elts[0]
    else:
        k = 2 ** (floor(log2(n)) - 1)
        return Stack(nest_stacks(elts[:n-k], n-k), 
                     nest_stacks(elts[n-k:], k))

def gen_random_expr(zs, n_objs, aexprs, k):
    envs = [{'z': z} for z in zs]
    objects = [Line, Rect]

    objs = []
    n_found = 0
    n_misses = 0
    while n_found < n_objs:
        f = choice(objects)
        args = [choice(aexprs) for t in f.in_types]
        e = f(*args)
        
        # check that e is well-formed
        outs = eval(e, envs)
        ok = sum(out is not None for out in outs)
        if ok >= k * len(zs):
            objs.append(e)
            n_found += 1
        else:
            n_misses += 1    
        print("Misses:", n_misses, end='\r')

    return nest_stacks(objs, n_objs)

def gen_random_exprs(zs, n_exprs, n_objs, aexprs, k):
    exprs = []
    for i in range(n_exprs):
        expr = gen_random_expr(zs, n_objs, aexprs, k)
        print('expr generated:', expr)
        exprs.append(expr)
    return exprs

if __name__ == '__main__':
    n_exprs = 10
    n_zs = 100
    abound = 1
    n_objs = 3

    zs = gen_zs(n_zs)
    g = Grammar(ops=[Plus, Minus, Times], 
                consts=([Z(i) for i in range(Z_SIZE)] + 
                        [Num(i) for i in range(Z_LO, Z_HI + 1)]))
    envs = [{'z':z} for z in zs]
    aexprs = [e for e, s in bottom_up_generator(abound, g, envs)]
    # for a in aexprs: print(a)

    # print(nest_stacks([Num(i) for i in range(10)], 10))
    exprs = gen_random_exprs(zs, n_exprs, n_objs, aexprs, n_zs * 0.5)
    
    for expr in exprs:
        for env in envs:
            try:
                viz(expr.eval(env), str(expr))
            except AssertionError:
                pass
