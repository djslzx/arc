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
import math
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
    # g_zbs = [Zb(Num(i)) for i in range(Z_SIZE)]
    g = Grammar(g.ops, g.consts + g_zns) # + g_zbs)

    # Randomly generate initial Zs
    zns = [gen_zn() for _ in range(len(exs))]
    zbs = [gen_zb() for _ in range(len(exs))]
    update_envs(exs, zns, zbs)

    f = None
    size = 1
    while size <= max_size:
        # optimize f up to `size`
        print("\nOptimizing f...")
        f, d, size = opt_f(g, exs, size, max_size)
        assert f is not None, f"Couldn't find an f with size <= {max_size}"
        zns = [env['z_n'] for env,_ in exs]
        print(f"\nd: {d}\nf: {f.pretty_print()}\nZn: {zns}")
        if d == 0:
            print(f"\n\nCompleted search at size {size}.\n\n")
            return f, zns

        # optimize z
        print("\nOptimizing Zn...")
        zns, score = opt_zns(f, zns, exs, samples)
        print(f"\nZn: {zns}, score: {score}")
        update_envs(exs, zns, zbs)
        if d == 0:
            print(f"\n\nCompleted search at size {size}.\n\n")
            return f, zns

    print(f"\nCompleted search at size {size}.\n")
    return f, zs

def opt_f(g, exs, bound, max_bound):
    """
    Optimize f wrt multiple examples: find f* = min_f sum_i d(f(s_i, z_i), x_i)
    """

    def score(dists):
        """
        Compute a score for a list of losses (distances)
        """
        return sum(dists)

    envs = [env for env,_ in exs]
    bmps = [bmp for _,bmp in exs]
    best_f = None
    best_d = None 
    for f, size in bottom_up_generator(max_bound, g, exs):
        if size > bound and best_f is not None:
            break
        if all(f.satisfies_invariants(env) for env in envs):
            outs = tuple(f.eval(env) for env in envs)
            if isinstance(outs[0], Bitmap):
                d = score(out.dist(bmp) for out, bmp in zip(outs, bmps))
                if d == 0:
                    return f, 0, size
                elif best_d is None or d < best_d:
                    best_f, best_d = f, d
    return best_f, best_d, size

def opt_zns(f, zs, exs, iters):
    """
    Optimize Z wrt multiple examples: find Z* = (z1, z2, ..., zn) where

      z_i = min_z d(f(s_i, z), x_i)

    ex: a list of examples (s_i, x_i) where s_i is an environment (including 'zn', 'zb') and x_i is a bitmap
    samples: max samples to randomly generate each z_i
    """
    def cost(f, z, ex):
        env, ans = ex
        # print(f"old_env={env}", end=", ")
        env['z_n'] = z
        # print(f"env={env}, ans={ans}, f={f}, dist={f.eval(env).dist(ans)}, sat={f.satisfies_invariants(env)}")
        return f.eval(env).dist(ans) if f.satisfies_invariants(env) else math.inf

    def best_neighbor(f, z, ex):
        def make_and_score_neighbor(z, i, d):
            n = z[:i] + [z[i] + d] + z[i+1:]
            # print(f"z={z}, cost(z)={cost(f, z, ex)}, n={n}, cost(n)={cost(f, n, ex)}")
            # print(f"z={z}, n={n}")
            return n, cost(f, n, ex)

        return min((make_and_score_neighbor(z, i, d)
                    for i in range(len(z)) 
                    for d in [-1, 1]), 
                   key=(lambda t: t[1]))

    def climb_hill(f, z, ex, max_iters):
        current = z
        current_cost = cost(f, z, ex)
        for _ in range(max_iters):
            n, n_cost = best_neighbor(f, current, ex)
            # print(f"c={current}, cost(c)={current_cost}, n={n}, cost(n)={n_cost}")
            if n_cost >= current_cost:
                # reached peak
                return current, current_cost
            current = n
            current_cost = n_cost

    zs_w_costs = [climb_hill(f, z, ex, iters) for (z, ex) in zip(zs, exs)]
    zs = [z for z,_ in zs_w_costs]
    cost = sum(c for _,c in zs_w_costs)
    return zs, cost

def test_opt_zns():
    f = Rect(Zn(Num(0)), Zn(Num(0)), 
             Zn(Num(1)), Zn(Num(2)))
    zs = [[0,1,0,0,0,0],]
    exs = [({}, # {'z_b': [False] * 6}, 
            Bitmap.from_img(['#___',
                             '#___',
                             '#___',
                             '#___',]))]
    out = opt_zns(f, zs, exs, iters=100)
    print(f"out={out}")

def test_learn():
    g = Grammar(
        ops=[Rect, Program, Plus, ], # Minus, Times, ] # If, Not, And, ]
        consts=[Num(0), Num(1), Num(2)])

    test_cases = [
        [
            ({}, Bitmap.from_img(['#___',
                                  '#___',
                                  '____',
                                  '____',])),
            ({}, Bitmap.from_img(['##__',
                                  '##__',
                                  '____',
                                  '____',])),
            ({}, Bitmap.from_img(['###_',
                                  '###_',
                                  '____',
                                  '____',])),
        ],
        # [
        #     ({}, Bitmap.from_img(['#___',
        #                           '____',
        #                           '____',
        #                           '____',])),
        #     ({}, Bitmap.from_img(['##__',
        #                           '##__',
        #                           '____',
        #                           '____',])),
        #     ({}, Bitmap.from_img(['###_',
        #                           '###_',
        #                           '###_',
        #                           '____',])),
        #     ({}, Bitmap.from_img(['####',
        #                           '####',
        #                           '####',
        #                           '####',])),
        # ],
        # [
        #     ({}, Bitmap.from_img(['#___',
        #                           '____',
        #                           '____',
        #                           '____',])),
        #     ({}, Bitmap.from_img(['#___',
        #                           '#___',
        #                           '____',
        #                           '____',])),
        #     ({}, Bitmap.from_img(['#___',
        #                           '#___',
        #                           '#___',
        #                           '____',])),
        # ],
        # [
        #     ({}, Bitmap.from_img(['#___',
        #                           '____',
        #                           '____',
        #                           '___#',])),
        #     ({}, Bitmap.from_img(['#___',
        #                           '#___',
        #                           '___#',
        #                           '___#',])),
        #     ({}, Bitmap.from_img(['#___',
        #                           '#__#',
        #                           '#__#',
        #                           '___#',])),
        # ],
        # [
        #     ({}, Rect(Num(0), Num(0), 
        #               Num(1), Num(1)).eval()),
        # ],
        # R(z1, z1, z1+1, z1+1)
        # [
        #     ({}, Rect(Num(0), Num(0), 
        #               Num(1), Num(1)).eval()),
        #     ({}, Rect(Num(1), Num(1), 
        #               Num(2), Num(2)).eval()),
        # ],
        # R(z1, z1, z1+1, z1+1)
        # [
        #     ({}, Rect(Num(0), Num(0), Num(1), Num(1))),
        #     ({}, Rect(Num(1), Num(1), Num(2), Num(2))),
        #     ({}, Rect(Num(3), Num(3), Num(4), Num(4))),
        # ],
        # R(z1, z1, z2, z3)
        # [
        #     ({}, Rect(Num(1), Num(1), 
        #               Num(4), Num(4)).eval()),
        #     ({}, Rect(Num(2), Num(2), 
        #               Num(3), Num(4)).eval()),
        # ],
        # R(x1, x2, x1+1, x2+1), R(1, 1, 2, 2)
        # [
        #     ({}, Program(Rect(Num(0), Num(0), 
        #                       Num(1), Num(1)),
        #                  Rect(Num(1), Num(1), 
        #                       Num(2), Num(2)))).eval(),
        #     ({}, Program(Rect(Num(1), Num(2), 
        #                       Num(2), Num(3)),
        #                  Rect(Num(1), Num(1), 
        #                       Num(2), Num(2)))).eval(),
        #     ({}, Program(Rect(Num(3), Num(1), 
        #                       Num(4), Num(2)),
        #                  Rect(Num(1), Num(1), 
        #                       Num(2), Num(2)))).eval(),
        # ],
        # [
        #     ({}, Program(Rect(Num(0), Num(1), 
        #                       Num(2), Num(3)),
        #                  Rect(Num(3), Num(2), 
        #                       Num(4), Num(4))).eval())
        # ],
        ## R(z0, z1, z2, z3), R(z3, z2, z4, z4)
        # [
        #     ({}, Program(Rect(Num(0), Num(1), 
        #                       Num(2), Num(3)),
        #                  Rect(Num(3), Num(2), 
        #                       Num(4), Num(4))).eval()),
        #     ({}, Program(Rect(Num(1), Num(2), 
        #                       Num(3), Num(3)),
        #                  Rect(Num(3), Num(3), 
        #                       Num(4), Num(4))).eval()),
        # ],
    ]

    for test_case in test_cases:
        start_time = time.time()
        print(f"\nTesting {[(env, p.pretty_print()) for env, p in test_case]}...")
        # exs = [(env, p.eval(env)) for env, p in test_case]
        f, zs = learn(g, test_case, max_size=20, samples=1000)
        end_time = time.time()
        fstr, used_zs = (f.pretty_print(), f.zs()) if f is not None else 'None', []
        print(f"\nSynthesized program:\t {fstr} \nused zs: {used_zs} \nZ: {zs} in {end_time - start_time}s")

if __name__ == '__main__':
    # test_opt_zns()
    test_learn()
