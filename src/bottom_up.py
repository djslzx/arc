import itertools
import time
import random
from pprint import pp
from grammar import *

VERBOSE = True

def eval(expr, xs):
    def handle(x):
        y = None
        try: y = expr.eval(x)
        except AssertionError: pass
        return y

    return tuple(handle(x) for x in xs)

def bottom_up(global_bound, grammar, exs):
    """
    global_bound: int. an upper bound on the size of expression
    exs: list of tuples of environment (the input) and desired output, such as [({'x': 5}, 6), ({'x': 1}, 2)]
    returns: either None if no program can be found that satisfies the input outputs, or the smallest such program. If a program `p` is returned, it should satisfy `all( p.eval(input) == output for input, output in exs )`
    """
    xs = [x for x, _ in exs]
    ys = tuple(y for _, y in exs)

    for expr, _ in bottom_up_generator(global_bound, grammar, xs):
        zs = eval(expr, xs)
        if ys == zs:
            return expr

    return None

def bottom_up_generator(global_bound, grammar, xs):
    """
    global_bound: int. an upper bound on the size of expression
    yields: sequence of programs, ordered by expression size, which are semantically distinct on the input examples
    """
    exprs = dict() # (type, size) => values
    seen = set()

    def add_expr(key, expr):
        """
        Add a expr to exprs[k] if not observationally equivalent to a pre-existing program.

        Use (output, type) pairs to address an issue with how Python thinks True == 1
        """
        ys = tuple((y, type(y)) for y in eval(expr, xs))
        if any(ys) and ys not in seen: # don't include `ys` if all entries are None
            seen.add(ys)
            if key not in exprs: exprs[key] = [expr]
            else: exprs[key].append(expr)    

    # Add all terminals to `exprs` as exprs of size 1
    for x in grammar.consts:
        yield x, 1
        add_expr((x.return_type, 1), x)

    # Generate all exprs of size `size` and add to `exprs`
    for size in range(1, global_bound+1):
        if VERBOSE: print(f"Generating exprs of size {size}... ", end='')
        # For each operator, determine how many args it needs and fill in using integer_partitions
        for op in grammar.ops:
            in_types = op.argument_types
            n_inputs = len(in_types)
            for partition in integer_partitions(size - 1 - n_inputs, n_inputs):
                for args in itertools.product(*[exprs.get((typ, size+1), []) 
                                                for typ, size in zip(in_types, partition)]):
                    expr = op(*args)
                    yield expr, size
                    add_expr((op.return_type, size), expr)
        if VERBOSE: print(f"|exprs| = {sum(len(l) for l in exprs.values())}")

def integer_partitions(target_value, number_of_arguments):
    """
    Returns all ways of summing up to `target_value` by adding `number_of_arguments` nonnegative integers
    """
    if target_value < 0:
        return []

    if number_of_arguments == 1:
        return [[target_value]]

    return [ [x1] + x2s
             for x1 in range(target_value + 1)
             for x2s in integer_partitions(target_value - x1, number_of_arguments - 1) ]

def test_bottom_up():
    g = Grammar(
        ops=[Program, Rect, Plus, Minus, Times, If, And, Not],
        consts=[Num(i) for i in range(5)] + [Zb(Num(i)) for i in range(Z_SIZE)] + [Zn(Num(i)) for i in range(Z_SIZE)],
    )

    # collection of input-output specifications
    test_cases = [
        [({}, 1)],
        [
            ({"z_n": [0, 0, 0, 0, 0, 1]}, Num(1).eval({})),
            ({"z_n": [0, 0, 0, 0, 0, 2]}, Num(2).eval({})),
            ({"z_n": [0, 0, 0, 0, 0, 3]}, Num(3).eval({})),
        ],
        [({"z_n": [0, 0, 0, 0, 0, 1]}, Num(1).eval({}))],
        [({"z_n": [0, 1, 2, 3, 4, 5]}, 
          Rect(Num(1), Num(1), 
               Num(3), Num(4)).eval({}))],
        [({"z_n": [3, 3, 4, 4, 0, 0]}, 
          Rect(Num(3), Num(3), 
               Num(4), Num(4)).eval({}))],
        [({"z_n": [0, 1, 2, 3, 4, 4]}, 
          Program(Rect(Num(1), Num(1), 
                       Num(2), Num(2)),
                  Rect(Num(2), Num(2), 
                       Num(3), Num(3))).eval({})),
         ({"z_n": [1, 2, 3, 4, 4, 4]}, 
          Program(Rect(Num(2), Num(2), 
                       Num(3), Num(3)),
                  Rect(Num(3), Num(3), 
                       Num(4), Num(4))).eval({}))
         ],
        [({"z_n": [0, 0, 1, 2, 3, 4]}, 
          Program(Rect(Num(1), Num(1), 
                       Num(2), Num(2)),
                  Rect(Num(3), Num(3), 
                       Num(4), Num(4))).eval({})),
         ({"z_n": [0, 0, 2, 3, 3, 4]},
          Program(Rect(Num(2), Num(2), 
                       Num(3), Num(3)),
                  Rect(Num(3), Num(3), 
                       Num(4), Num(4))).eval({})),],
    ]
    bound = 25
    for test_case in test_cases:
        envs = [env for env,_ in test_case]
        for env in envs:
            if 'z_n' not in env:
                env['z_n'] = list(range(Z_SIZE))
            if 'z_b' not in env:
                env['z_b'] = [i % 2 == 0 for i in range(Z_SIZE)]
        start_time = time.time()
        print(f"\nTesting {test_case}...")
        expr = bottom_up(bound, g, test_case)
        print(f"\nSynthesized program:\t {expr.pretty_print() if expr is not None else 'None'} in {time.time() - start_time} seconds")

    # print(" [+] bottom-up synthesis passes tests")

if __name__ == "__main__":
    test_bottom_up()
