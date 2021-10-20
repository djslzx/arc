"""
Learn a program `f` that maps envs (including Zn, Zb's) to bitmaps

Inputs: {x_i, y_i | i < n} where x_i is an env and y_i is a bitmap
Output: A program P s.t. for all i, P.eval(x_i) ~= y_i.  
        More concretely, for all i, dist(P.eval(x_i), y_i) < threshold 
        where 'threshold' is a hyperparameter

Strategy: jointly optimize `z` and `f`
"""

import random
import itertools
from grammar import *
from bottom_up import *


def gen_zb():
    return [bool(random.randint(0,1)) for _ in range(Z_SIZE)]

def gen_zn():
    return [random.randint(Z_LO, Z_HI) for _ in range(Z_SIZE)]

def learn(g, exs, max_size, samples):
    """
    Jointly optimize f and Z wrt multiple examples {(s_i,x_i)}_i,
    where each s_i is an environment and each x_i is a bitmap

    max_size: max number of rounds to run, where one round optimizes f, then z, then f again
    samples: max number of samples to take when optimizing z
    """
    def update_envs(exs, zns, zbs):
        """
        Update z_n, z_b in each env in envs
        """
        envs = [env for env,_ in exs]
        for env, zn, zb  in zip(envs, zns, zbs):
            env['z_n'] = zn
            env['z_b'] = zb

    # Add Zs to grammar
    g_zns = [Zn(Num(i)) for i in range(Z_SIZE)]
    g_zbs = [Zb(Num(i)) for i in range(Z_SIZE)]
    g = Grammar(g.ops, g.consts + g_zns + g_zbs)

    # Randomly generate initial Zs
    zns = [gen_zn() for _ in range(len(exs))]
    zbs = [gen_zb() for _ in range(len(exs))]
    update_envs(exs, zns, zbs)

    f = None
    for size in range(1, max_size):

        # optimize f up to `size`
        print("\nOptimizing f...")
        f, d = opt_f(g, exs, size)
        if f is None: 
            print("Could not find a valid f; increasing size...")
            continue
        zs = [(env['z_n'], env['z_b']) for env,_ in exs]
        print(f"\nd: {d}\nf: {f.pretty_print()}\nZ: {zs}")
        if d == 0:
            print(f"\n\nCompleted search at size {size}.\n\n")
            return f, zs

        # optimize z
        print("\nOptimizing z...")
        zns, d = opt_zns(f, exs, samples)
        print(f"\nd: {d}\nZ: {zs}")
        update_envs(exs, zns, zbs)
        if d == 0:
            print(f"\n\nCompleted search at size {size}.\n\n")
            return f, zs

    print(f"\n\nCompleted search at size {size}.\n\n")
    return f, zs

def opt_f(g, exs, bound):
    """
    Optimize f wrt multiple examples: find f* = min_f sum_i d(f(s_i, z_i), x_i)

    ops: a list of nodes in AST, e.g. [Times, If, ...]
    consts: a list of leaves in AST
    exs: a list of examples (s_i, x_i) where s_i is an environment and x_i is a bitmap
    """

    def score(dists):
        """
        Compute a score for a list of losses (distances)
        """
        return sum(dists)

    gen = bottom_up_generator(bound, g, exs)
    envs = [env for env,_ in exs]
    bmps = [bmp for _,bmp in exs]
    best_f = None
    best_d = None 
    for f in gen:
        if all(f.satisfies_invariants(env) for env in envs):
            outs = tuple(f.eval(env) for env in envs)
            if isinstance(outs[0], Bitmap):
                d = score(out.dist(bmp) for out, bmp in zip(outs, bmps))
                if d == 0:
                    return f, 0
                elif best_d is None or d < best_d:
                    best_f, best_d = f, d
    return best_f, best_d

def opt_zns(f, exs, samples):
    """
    Optimize Z wrt multiple examples: find Z* = (z1, z2, ..., zn) where

      z_i = min_z d(f(s_i, z), x_i)

    ex: a list of examples (s_i, x_i) where s_i is an environment (including 'zn', 'zb') and x_i is a bitmap
    samples: max samples to randomly generate each z_i
    """
    def opt_zn(f, ex, samples):
        env, ans = ex
        best_zn = None
        best_d = None
        for _ in range(samples):
            zn = gen_zn()
            env['z_n'] = zn
            if f.satisfies_invariants(env):
                d = f.eval(env).dist(ans)
                if d == 0:
                    return zn, 0
                elif best_zn is None or d < best_d:
                    best_zn, best_d = zn, d
        return best_zn, best_d

    zns_ds = [opt_zn(f, ex, samples) for ex in exs]
    zns = [zn for zn,_ in zns_ds]
    d = sum(d for _,d in zns_ds)
    return zns, d

def test_learn():
    g = Grammar(
        ops=[Point, Rect, Program, Plus, ], # Minus, Times, ] # If, Not, And, ]
        consts=[Num(0), Num(1), Num(2)])

    test_cases = [
        [
            ({}, Rect(Point(Num(0), Num(0)), Point(Num(1), Num(1)))),
        ],
        # R(z1, z1, z2, z2)
        # [
        #     ({}, Rect(Point(Num(0), Num(0)), Point(Num(1), Num(1)))),
        #     ({}, Rect(Point(Num(1), Num(1)), Point(Num(2), Num(2)))),
        # ],
        # R(z1, z1, z2, z3)
        # [
        #     ({}, Rect(Point(Num(0), Num(0)), Point(Num(1), Num(1)))),
        #     ({}, Rect(Point(Num(1), Num(1)), Point(Num(2), Num(2)))),
        #     ({}, Rect(Point(Num(3), Num(3)), Point(Num(4), Num(4)))),
        # ],
        # R(z1, z1, z2, z3)
        # [
        #     ({}, Rect(Point(Num(1), Num(1)), 
        #               Point(Num(4), Num(4)))),
        #     ({}, Rect(Point(Num(2), Num(2)), 
        #               Point(Num(3), Num(4)))),
        # ],
        # R(x1, x2, x1+1, x2+1), R(1, 1, 2, 2)
        # [
        #     ({}, Program(Rect(Point(Num(0), Num(0)), 
        #                       Point(Num(1), Num(1))),
        #                  Rect(Point(Num(1), Num(1)), 
        #                       Point(Num(2), Num(2))))),
        #     ({}, Program(Rect(Point(Num(1), Num(2)), 
        #                       Point(Num(2), Num(3))),
        #                  Rect(Point(Num(1), Num(1)), 
        #                       Point(Num(2), Num(2))))),
        #     ({}, Program(Rect(Point(Num(3), Num(1)), 
        #                       Point(Num(4), Num(2))),
        #                  Rect(Point(Num(1), Num(1)), 
        #                       Point(Num(2), Num(2))))),
        # ],
        # [
        #     ({}, Program(Rect(Point(Num(0), Num(1)), 
        #                       Point(Num(2), Num(3))),
        #                  Rect(Point(Num(3), Num(2)), 
        #                       Point(Num(4), Num(4)))))
        # ],
        ## R(z0, z1, z2, z3), R(z3, z2, z4, z4)
        # [
        #     ({}, Program(Rect(Point(Num(0), Num(1)), 
        #                       Point(Num(2), Num(3))),
        #                  Rect(Point(Num(3), Num(2)), 
        #                       Point(Num(4), Num(4))))),
        #     ({}, Program(Rect(Point(Num(1), Num(2)), 
        #                       Point(Num(3), Num(3))),
        #                  Rect(Point(Num(3), Num(3)), 
        #                       Point(Num(4), Num(4))))),
        # ],
    ]

    for test_case in test_cases:
        start_time = time.time()
        print(f"\nTesting {[(env, p.pretty_print()) for env, p in test_case]}...")
        exs = [(env, p.eval(env)) for env, p in test_case]
        f, zs = learn(g, exs, max_size=20, samples=100)
        print(f"\nSynthesized program:\t {f.pretty_print() if f is not None else 'None'}, \nZ: {zs} in {time.time() - start_time} seconds")

if __name__ == '__main__':
    test_learn()
