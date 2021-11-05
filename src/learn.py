"""
Learn a program `f` that maps envs (including Zn, Zb's) to bitmaps

Inputs: {x_i, y_i | i < n} where x_i is an env and y_i is a bitmap
Output: A program P s.t. for all i, P.eval(x_i) ~= y_i.  
        More concretely, for all i, dist(P.eval(x_i), y_i) < threshold 
        where 'threshold' is a hyperparameter

Strategy: jointly optimize `z` and `f`
"""

import random
import torch as T
from util import *
from grammar import *
from bottom_up import *

def gen_zb():
    return [bool(random.randint(0,1)) for _ in range(Z_SIZE)]

def gen_zn():
    return [random.randint(Z_LO, Z_HI) for _ in range(Z_SIZE)]

def render(f, envs, cost):
    zns = [env['z_n'] for env in envs]
    print(f"\nf={f.pretty_print()}, Z={zns}, cost={cost}")
    print("exs:")
    for env in envs:
        print(f"in={env['z_n']}, out=")
        print(f.eval(env).pretty_print())

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
        f, cost, size = opt_f(g, exs, size, max_size)
        assert f is not None, f"Couldn't find an f with size <= {max_size}"
        render(f, [env for env,_ in exs], cost)
        if cost == 0: break

        # optimize z
        print("\nOptimizing Zn...")
        zns, cost = opt_zns(f, zns, exs, samples)
        update_envs(exs, zns, zbs)
        render(f, [env for env,_ in exs], cost)
        if cost == 0: break

    print(f"\nCompleted search at size {size}.\n")
    return f, zns

def opt_f(g, exs, bound, max_bound):
    """
    Optimize f wrt multiple examples: find f* = min_f sum_i d(f(s_i, z_i), x_i)
    """
    envs = [env for env,_ in exs]
    bmps = [bmp for _,bmp in exs]
    best_f = None
    best_d = None 
    for f, size in bottom_up_generator(max_bound, g, envs):
        if size > bound and best_f is not None:
            break
        # TODO: eval once and use this for satisfies_invariants instead of evaluating twice
        if f.return_type == "Bitmap":
            ys = eval(f, envs)
            # print("ys", [y.dim for y in ys if y is not None], "bmps", [bmp.dim for bmp in bmps])
            if any(ys):
                d = sum_sq(bmp.dist(y) for y, bmp in zip(ys, bmps))
                if d == 0:
                    return f, 0, size
                elif best_d is None or d < best_d:
                    best_f, best_d = f, d

    return best_f, best_d, size

def cost(f, z, ex):
    """Hill climbing cost function for f, z wrt an example"""
    env, ans = ex
    env['z_n'] = z

    # TODO: penalize use of lots of components of z
    n_zs = max(len(f.zs()), 1)
    try:
        return n_zs * f.eval(env).dist(ans)
    except AssertionError:
        return T.inf

def opt_zn(f, z, ex, mask, iters):

    def zn_cost(z):
        return cost(f, z, ex)

    def best_neighbor(z, deltas=[-1, 1]):
        """
        Choose best neighbor of z,
          z* = argmin_{n in N(z)} cost(n)
        where the neighborhood of z is defined as
          N(z) = {v | len(v) = len(z), v differs from x at exactly one component}
        """
        # Regenerate any z's that aren't currently used
        for i, b in enumerate(mask):
            if not b:
                z[i] = random.randint(Z_LO, Z_HI)
            
        neighbors = [z[:i] + [z[i] + d] + z[i+1:]
                     for i, b in enumerate(mask) if b
                     for d in deltas]

        return min(((n, zn_cost(n)) for n in neighbors), 
                   key=(lambda t: t[1]))

    def climb_hill():
        current = z
        current_cost = zn_cost(z)
        for i in range(iters):
            n, n_cost = best_neighbor(current)
            if n_cost >= current_cost: # reached peak
                return current, current_cost
            current = n
            current_cost = n_cost
        return current, current_cost

    def random_choice():
        current = z
        current_cost = zn_cost(z)
        for _ in range(iters):
            n = gen_zn()
            n_cost = zn_cost(n)
            if n_cost == 0: return n, n_cost
            elif n_cost < current_cost:
                current, current_cost = n, n_cost
        return current, current_cost

    return climb_hill()
    # return random_choice()

def opt_zns(f, zs, exs, iters):
    """
    Optimize Z wrt multiple examples: find Z* = (z1, z2, ..., zn) where

      z_i = min_z d(f(s_i, z), x_i)

    ex: a list of examples (s_i, x_i) where s_i is an environment (including 'zn', 'zb') and x_i is a bitmap
    samples: max samples to randomly generate each z_i
    """
    mask = [i in f.zs() for i in range(Z_SIZE)]
    if not any(mask): # If f uses no components of z, just randomly generate a new z
        print("Empty mask, randomizing...")
        zs = [gen_zn() for _ in range(len(zs))]
        c = sum_sq(cost(f, z, ex) for z, ex in zip(zs, exs))
        return zs, c
    zs_w_costs = [opt_zn(f, z, ex, mask, iters) for (z, ex) in zip(zs, exs)]
    zs = [z for z,_ in zs_w_costs]
    c = sum_sq(c for _,c in zs_w_costs)
    return zs, c

def test_opt_zns():
    # f = Rect(Zn(Num(0)), Zn(Num(0)), 
    #          Zn(Num(1)), Zn(Num(2)))
    # zs = [[0,1,0,0,0,0],]
    # exs = [({}, # {'z_b': [False] * 6}, 
    #         Bitmap.from_img(['#___',
    #                          '#___',
    #                          '#___',
    #                          '#___',]))]
    # out = opt_zns(f, zs, exs, iters=100)
    # print(f"out={out}")

    f = Rect(Num(0), Num(0), 
             Zn(Num(0)), Zn(Num(0)))
    zs = [[4, 0, 0, 4, 0, 4], 
          [4, 0, 4, 4, 1, 0], 
          [1, 0, 4, 1, 3, 2], 
          [1, 1, 1, 3, 2, 1]]
    exs = [
            ({}, Bitmap.from_img(['#___',
                                  '____',
                                  '____',
                                  '____',])),
            ({}, Bitmap.from_img(['##__',
                                  '##__',
                                  '____',
                                  '____',])),
            ({}, Bitmap.from_img(['###_',
                                  '###_',
                                  '###_',
                                  '____',])),
            ({}, Bitmap.from_img(['####',
                                  '####',
                                  '####',
                                  '####',])),
    ]
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
        [
            ({}, Bitmap.from_img(['#___',
                                  '____',
                                  '____',
                                  '____',])),
            ({}, Bitmap.from_img(['##__',
                                  '##__',
                                  '____',
                                  '____',])),
            ({}, Bitmap.from_img(['###_',
                                  '###_',
                                  '###_',
                                  '____',])),
            ({}, Bitmap.from_img(['####',
                                  '####',
                                  '####',
                                  '####',])),
        ],
        [
            ({}, Bitmap.from_img(['#___',
                                  '____',
                                  '____',
                                  '____',])),
            ({}, Bitmap.from_img(['#___',
                                  '#___',
                                  '____',
                                  '____',])),
            ({}, Bitmap.from_img(['#___',
                                  '#___',
                                  '#___',
                                  '____',])),
        ],
        # [
        #     ({}, Bitmap.from_img(['##__',
        #                           '##__',
        #                           '####',
        #                           '####',])),
        #     ({}, Bitmap.from_img(['_##_',
        #                           '_##_',
        #                           '####',
        #                           '####',])),
        #     ({}, Bitmap.from_img(['__##',
        #                           '__##',
        #                           '####',
        #                           '####',])),
        # ],
        [
            ({}, Bitmap.from_img(['#___',
                                  '____',
                                  '____',
                                  '____',])),
            ({}, Bitmap.from_img(['___#',
                                  '____',
                                  '____',
                                  '____',])),
            ({}, Bitmap.from_img(['____',
                                  '____',
                                  '____',
                                  '#___',])),
            ({}, Bitmap.from_img(['____',
                                  '____',
                                  '____',
                                  '___#',])),
        ],
        # [
        #     ({}, Bitmap.from_img(['##__',
        #                           '##__',
        #                           '____',
        #                           '____',])),

        #     ({}, Bitmap.from_img(['__##',
        #                           '__##',
        #                           '____',
        #                           '____',])),

        #     ({}, Bitmap.from_img(['____',
        #                           '____',
        #                           '##__',
        #                           '##__',])),

        #     ({}, Bitmap.from_img(['____',
        #                           '____',
        #                           '__##',
        #                           '__##',])),
        # ],
        # |P[R(0,0,1,a), R(3,4-a,4,4)]| = 13
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
        #     ({}, Bitmap.from_img(['#___',
        #                           '____',
        #                           '____',
        #                           '____',])),
        # ],
        # [
        #     ({}, Bitmap.from_img(['#___',
        #                           '____',
        #                           '____',
        #                           '____',])),
        #     ({}, Bitmap.from_img(['____',
        #                           '_#__',
        #                           '____',
        #                           '____',])),
        # ],
        # [
        #     ({}, Bitmap.from_img(['#___',
        #                           '____',
        #                           '____',
        #                           '____',])),
        #     ({}, Bitmap.from_img(['____',
        #                           '_#__',
        #                           '____',
        #                           '____',])),
        #     ({}, Bitmap.from_img(['____',
        #                           '____',
        #                           '__#_',
        #                           '____',])),
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
        print("\nTesting ")
        for i, (env, p) in enumerate(test_case):
            print(f"Example {i}: env={env}, img=")
            print(p.pretty_print())
        # exs = [(env, p.eval(env)) for env, p in test_case]
        f, zs = learn(g, test_case, max_size=20, samples=1000)
        end_time = time.time()
        fstr, used_zs = (f.pretty_print(), f.zs()) if f is not None else ('None', [])
        print(f"Synthesized program:\t {fstr} used zs: {used_zs} \nZ: {zs} in {end_time - start_time}s")

if __name__ == '__main__':
    # test_opt_zns()
    test_learn()
