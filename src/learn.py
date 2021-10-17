"""
Learn a program `f` that maps envs (including Zn, Zb's) to bitmaps

Inputs: {x_i, y_i | i < n} where x_i is an env and y_i is a bitmap
Output: A program P s.t. for all i, P.eval(x_i) ~= y_i.  
        More concretely, for all i, dist(P.eval(x_i), y_i) < threshold 
        where 'threshold' is a hyperparameter

Strategy: jointly optimize `z` and `f`
"""

import random
from grammar import *
from bottom_up import *

def learn(ops, consts, exs, iters, bound):
    """
    Jointly optimize f and Z wrt multiple examples {(s_i,x_i)}_i,
    where each s_i is an environment and each x_i is a bitmap

    iters: max number of iterations to run; one iteration optimizes f, then z
    bound: max size of f (bottom-up enumeration bound)
    """
    def update_envs(exs, zs):
        """
        Update z_n, z_b in each env in envs
        """
        for (env, _), (zn, zb) in zip(exs, zs):
            env['z_n'] = zn
            env['z_b'] = zb

    for _ in range(iters):
        # optimize f
        f, d = opt_f(ops, consts, exs, bound)
        zs = [(env['z_n'], env['z_b']) for env,_ in exs]
        print(f"d: {d}\nf: {f.pretty_print()}\nZ: {zs}")
        if d == 0:
            return f, zs

        # optimize z
        zs, d = opt_zs(f, exs, iters)
        print(f"d: {d}, Z: {zs}")
        update_envs(exs, zs)
        if d == 0:
            return f, zs

    return f, zs

def opt_f(ops, consts, exs, bound):
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

    gen = bottom_up_generator(bound, ops, consts, exs)
    envs = [env for env,_ in exs]
    bmps = [bmp for _,bmp in exs]
    best_expr = None
    best_d = None
    for expr in gen:
        if all(expr.satisfies_invariants(env) for env in envs):
            outs = tuple(expr.eval(env) for env in envs)
            if isinstance(outs[0], Bitmap):
                d = score(out.dist(bmp) for out, bmp in zip(outs, bmps))
                if d == 0:
                    return expr, 0
                elif best_d is None or d < best_d:
                    best_d, best_expr = d, expr
    return best_expr, best_d

def opt_z(f, ex, iters):
    """
    Optimize z wrt a single example: find z* = min_z d(f(s, z), x)

    ex: an example (s,x) where s is an environment (including 'zn', 'zb') and x is a bitmap
    iters: max iters to randomly generate z
    """
    env, ans = ex
    best_zs = None
    best_d = None
    for _ in range(iters):
        s = env
        zn, zb = gen_zn(), gen_zb()
        s["z_n"] = zn
        s["z_b"] = zb
        d = f.eval(s).dist(ans)
        if f.satisfies_invariants(s) and \
           (best_d is None or d < best_d):
            if best_d == 0:
                return best_zs, best_d
            best_zs, best_d = (zn, zb), d
    return best_zs, best_d

def opt_zs(f, exs, iters):
    """
    Optimize z wrt multiple examples
    """
    zs_and_ds = [opt_z(f, ex, iters) for ex in exs]
    zs = [z for z,_ in zs_and_ds]
    d = sum(d for _,d in zs_and_ds)
    return zs, d
            
def test_learn():
    ops = [Point, Rect, Program, ] # Plus, Minus, Times, If, Not, And, ]
    consts = []

    test_cases = [
        # [({"z_n": [1]*Z_SIZE}, Point(Num(1), Num(1))),
        #  ({"z_n": [2]*Z_SIZE}, Point(Num(2),Num(2)))],
        [
            ({}, Rect(Point(Num(0), Num(0)), Point(Num(1), Num(1)))),
        ],
        [
            ({}, Rect(Point(Num(0), Num(0)), Point(Num(1), Num(1)))),
            ({}, Rect(Point(Num(1), Num(1)), Point(Num(2), Num(2)))),
        ],
        [
            ({}, Rect(Point(Num(0), Num(0)), Point(Num(1), Num(1)))),
            ({}, Rect(Point(Num(1), Num(1)), Point(Num(2), Num(2)))),
            ({}, Rect(Point(Num(3), Num(3)), Point(Num(4), Num(4)))),
        ],
        # [({}, Rect(Point(Num(1), Num(1)), 
        #            Point(Num(1), Num(1))))],
        # [
        #     ({}, Rect(Point(Num(1), Num(1)), 
        #               Point(Num(4), Num(4)))),
        #     ({}, Rect(Point(Num(2), Num(2)), 
        #               Point(Num(3), Num(5)))),
        #  ],

        # [({}, Rect(Point(Num(1), Num(1)), 
        #            Point(Num(5), Num(6))))],
        # [({}, Program(Rect(Point(Num(0), Num(1)), 
        #                    Point(Num(3), Num(4))),
        #               Rect(Point(Num(5), Num(6)), 
        #                    Point(Num(7), Num(7)))))],
    ]

    for test_case in test_cases:
        print(f"Testing {[(env, p.pretty_print()) for env, p in test_case]}...")
        exs = [(env, p.eval(env)) for env, p in test_case]
        f, zs = learn(ops, consts, exs, iters=100, bound=8)


if __name__ == '__main__':
    test_learn()
