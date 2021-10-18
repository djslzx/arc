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

def added_env(env, zn, zb):
    """Return an env to which zn and zb have been added"""
    s = env
    s["z_n"] = zn
    s["z_b"] = zb
    return s

def learn(ops, consts, exs, rounds, bound, iters):
    """
    Jointly optimize f and Z wrt multiple examples {(s_i,x_i)}_i,
    where each s_i is an environment and each x_i is a bitmap

    rounds: max number of `rounds` to run, where one round optimizes f, then z
    bound: max size of f (bottom-up enumeration bound)
    iters: max number of iterations to run when optimizing z
    """
    def update_envs(exs, zs):
        """
        Update z_n, z_b in each env in envs
        """
        # try:
        for (env, _), (zn, zb) in zip(exs, zs):
            env['z_n'] = zn
            env['z_b'] = zb
        # except ValueError as err:
        #     assert False, f"err: {err}, exs={exs}, zs={zs}"

    f = None
    for _ in range(rounds):

        # optimize f
        print("\nOptimizing f...")
        f, d = opt_f(ops, consts, exs, bound)
        zs = [(env['z_n'], env['z_b']) for env,_ in exs]
        print(f"d: {d}\nf: {f.pretty_print()}\nZ: {zs}")
        if d == 0:
            return f, zs

        # optimize z
        print("\nOptimizing z...")
        zs, d = opt_zs(f, exs, iters)
        print(f"d: {d}\nZ: {zs}")
        update_envs(exs, zs)
        if d == 0:
            return f, zs

    return f, zs

def opt_f(ops, consts, exs, bound, f0=None):
    """
    Optimize f wrt multiple examples: find f* = min_f sum_i d(f(s_i, z_i), x_i)

    ops: a list of nodes in AST, e.g. [Times, If, ...]
    consts: a list of leaves in AST
v    exs: a list of examples (s_i, x_i) where s_i is an environment and x_i is a bitmap
    """

    def score(dists):
        """
        Compute a score for a list of losses (distances)
        """
        return sum(dists)

    gen = bottom_up_generator(bound, ops, consts, exs)
    if f0 is not None:
        gen = itertools.chain([f0], gen)
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
                    best_d, best_f = d, f
    return best_f, best_d

def opt_z(f, ex, iters, z0=None):
    """
    Optimize z wrt a single example: find z* = min_z d(f(s, z), x)

    ex: an example (s,x) where s is an environment (including 'zn', 'zb') and x is a bitmap
    iters: max iters to randomly generate z
    """
    env, ans = ex
    best_z = z0
    best_d = None if z0 is None else f.eval(added_env(env, *z0)).dist(ans)
    for _ in range(iters):
        zn, zb = gen_zn(), gen_zb()
        s = added_env(env, zn, zb)
        if f.satisfies_invariants(s):
            d = f.eval(s).dist(ans)
            if d == 0:
                return (zn, zb), 0
            elif best_z is None or d < best_d:
                best_z, best_d = (zn, zb), d
    return best_z, best_d

def opt_zs(f, exs, iters, zs0=None):
    """
    Optimize zs for multiple examples
    """
    zs_and_ds = \
        [opt_z(f, ex, iters) for ex in exs] \
        if zs0 is None \
        else [opt_z(f, ex, iters, z0) for ex, z0 in zip(exs, zs0)]
    print(f"zs and ds: {zs_and_ds}")
    zs = [z for z,_ in zs_and_ds]
    d = sum(d for _,d in zs_and_ds)
    return zs, d
            
def test_learn():
    ops = [Point, Rect, Program, Plus, Minus, Times, ] # If, Not, And, ]
    consts = []

    test_cases = [
        # [
        #     ({}, Rect(Point(Num(0), Num(0)), Point(Num(1), Num(1)))),
        # ],
        [
            ({}, Rect(Point(Num(0), Num(0)), Point(Num(1), Num(1)))),
            ({}, Rect(Point(Num(1), Num(1)), Point(Num(2), Num(2)))),
        ],
        # [
        #     ({}, Rect(Point(Num(0), Num(0)), Point(Num(1), Num(1)))),
        #     ({}, Rect(Point(Num(1), Num(1)), Point(Num(2), Num(2)))),
        #     ({}, Rect(Point(Num(3), Num(3)), Point(Num(4), Num(4)))),
        # ],
        # [({}, Rect(Point(Num(1), Num(1)), 
        #            Point(Num(1), Num(1))))],
        # [
        #     ({}, Rect(Point(Num(1), Num(1)), 
        #               Point(Num(4), Num(4)))),
        #     ({}, Rect(Point(Num(2), Num(2)), 
        #               Point(Num(3), Num(4)))),
        #  ],
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
    ]

    for test_case in test_cases:
        start_time = time.time()
        print(f"\nTesting {[(env, p.pretty_print()) for env, p in test_case]}...")
        exs = [(env, p.eval(env)) for env, p in test_case]
        f, zs = learn(ops, consts, exs, rounds=10, bound=10, iters=50)
        print(f"\nSynthesized program:\t {f.pretty_print() if f is not None else 'None'}, \nZ: {zs} in {time.time() - start_time} seconds")

if __name__ == '__main__':
    test_learn()
