import itertools
import time
import random
import torch as T
from grammar import *

VERBOSE = True

def eval(expr, envs):
    def handle(env):
        try: 
            return expr.eval(env)
        except AssertionError: 
            return None

    return tuple(handle(env) for env in envs)

def ret_type(x):
    if isinstance(x, T.Tensor): return 'bitmap'
    elif isinstance(x, int):    return 'int'
    elif isinstance(x, bool):   return 'bool'
    else:
        assert False, "unexpected return type"

def bottom_up(global_bound, grammar, exs):
    """
    global_bound: int. an upper bound on the size of expression
    exs: list of tuples of environment (the input) and desired output, such as [({'x': 5}, 6), ({'x': 1}, 2)]
    returns: either None if no program can be found that satisfies the input outputs, or the smallest such program. If a program `p` is returned, it should satisfy `all( p.eval(input) == output for input, output in exs )`
    """
    envs = [x for x, _ in exs]
    expected_outs = tuple(y for _, y in exs)
    return_type = ret_type(expected_outs[0])

    # print("Expected return type:", return_type)
    for expr, _ in bottom_up_generator(global_bound, grammar, envs):
        # print(expr, "expr return type:", expr.out_type)
        if expr.out_type == return_type:
            actual_outs = eval(expr, envs)
            if return_type == 'bitmap':
                if all(actual is not None and T.equal(actual, expected) 
                       for actual, expected in zip(actual_outs, expected_outs)):
                    return expr
            elif expected_outs == actual_outs: return expr
    return None

def bottom_up_generator(global_bound, grammar, envs):
    """
    global_bound: int. an upper bound on the size of expression
    yields: sequence of programs, ordered by expression size, which are semantically distinct on the input examples
    """
    exprs = dict() # (type, size) => values
    seen = set()

    def eval_expr(expr):
        return tuple((y, type(y)) for y in eval(expr, envs))        

    def new_expr(expr):
        ys = eval_expr(expr)
        is_new = ys not in seen and any(y is not None for y,_ in ys)
        seen.add(ys)
        return is_new

    def add_expr(key, expr):
        """
        Add a expr to exprs[k] if not observationally equivalent to a pre-existing program.

        Use (output, type) pairs to address an issue with how Python thinks True == 1
        """
        if key not in exprs: 
            exprs[key] = [expr]
        else: 
            exprs[key].append(expr)    

    # Add all terminals to `exprs` as exprs of size 1
    for x in grammar.consts:
        yield x, 1
        seen.add(eval_expr(x))
        add_expr((x.out_type, 1), x)

    # Generate all exprs of size `size` and add to `exprs`
    for size in range(1, global_bound+1):
        if VERBOSE: print(f"Generating exprs of size {size}... ", end='')
        # For each operator, determine how many args it needs and fill in using integer_partitions
        for op in grammar.ops:
            in_types = op.in_types
            n_inputs = len(in_types)
            for partition in integer_partitions(size - 1 - n_inputs, n_inputs):
                for args in itertools.product(*[exprs.get((typ, size+1), []) 
                                                for typ, size in zip(in_types, partition)]):
                    expr = op(*args)
                    if new_expr(expr):
                        yield expr, size
                        add_expr((op.out_type, size), expr)
                    
        if VERBOSE: 
            print(f"|exprs| = {sum(len(l) for l in exprs.values())}")
            # print("exprs:")
            # for k,v in exprs.items():
            #     print(k)
            #     for x in v:
            #         print("  ", x)

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
    grammar = Grammar(
        ops=[Join, CornerRect, CornerLine, Point, Plus, Minus, Times, If, And, Not],
        consts=[Num(i) for i in range(LIB_SIZE)] + [Z(i) for i in range(LIB_SIZE)],
    )

    # collection of input-output specifications
    test_cases = [
        [({'z':[0,1,2,3,4,5]}, 1)],
        [
            ({"z": [0, 0, 0, 0, 0, 1]}, Num(1).eval({})),
            ({"z": [0, 0, 0, 0, 0, 2]}, Num(2).eval({})),
            ({"z": [0, 0, 0, 0, 0, 3]}, Num(3).eval({})),
        ],
        [({"z": [0, 0, 0, 0, 0, 1]}, Num(1).eval({}))],
        [({"z": [0, 1, 2, 3, 4, 5]},
          CornerRect(Num(1), Num(1),
                     Num(3), Num(4)).eval({}))],
        [({"z": [3, 3, 4, 4, 0, 0]},
          CornerRect(Num(3), Num(3),
                     Num(4), Num(4)).eval({}))],
        # U((R Z1 Z1 Z2 Z2) (R Z2 Z2 Z3 Z3))
        [({"z": [0, 1, 2, 3, 4, 4]},
          Join(CornerRect(Num(1), Num(1),
                          Num(2), Num(2)),
               CornerRect(Num(2), Num(2),
                          Num(3), Num(3))).eval({})),
         ({"z": [1, 2, 3, 4, 4, 4]},
          Join(CornerRect(Num(2), Num(2),
                          Num(3), Num(3)),
               CornerRect(Num(3), Num(3),
                          Num(4), Num(4))).eval({}))
         ],
        # [({"z": [0, 0, 1, 2, 3, 4]}, 
        #   Union(Rect(Num(1), Num(1), 
        #              Num(2), Num(2)),
        #         Rect(Num(3), Num(3), 
        #              Num(4), Num(4))).eval({})),
        #  ({"z": [0, 0, 2, 3, 3, 4]},
        #   Union(Rect(Num(2), Num(2), 
        #              Num(3), Num(3)),
        #         Rect(Num(3), Num(3), 
        #              Num(4), Num(4))).eval({})),],
    ]
    bound = 25
    for test_case in test_cases:
        envs = [env for env,_ in test_case]
        start_time = time.time()
        print(f"\nTesting {test_case}...")
        expr = bottom_up(bound, grammar, test_case)
        print(f"\nSynthesized program:\t {expr} in {time.time() - start_time} seconds")

    # print(" [+] bottom-up synthesis passes tests")

def test_bottom_up_tensor():
    grammar = Grammar(
        ops=[Join, CornerRect, Plus, Minus, Times, Apply, HFlip, VFlip, Translate],
        consts= [Z(i) for i in range(LIB_SIZE)] # [Num(i) for i in range(5)]
    )

    test_cases = [
        [
            ({"z": T.tensor([0, 0, 0, 0, 0, 1])}, 1),
            ({"z": T.tensor([0, 0, 0, 0, 0, 2])}, 2),
            ({"z": T.tensor([0, 0, 0, 0, 0, 3])}, 3),
        ],
        [
            ({"z": T.tensor([0, 1, 2, 3, 4, 5])},
             CornerRect(Num(1), Num(1),
                        Num(3), Num(4)).eval({}))
        ],
    ]

    bound = 5
    for test_case in test_cases:
        envs = [env for env,_ in test_case]
        for env in envs:
            if 'z' not in env:
                env['z'] = T.rand(LIB_SIZE)

        start_time = time.time()
        print(f"\nTesting {test_case}...")
        expr = bottom_up(bound, grammar, test_case)
        print(f"\nSynthesized program:\t {expr} in {time.time() - start_time} seconds")
    

if __name__ == "__main__":
    # test_bottom_up_tensor()
    test_bottom_up()
