import itertools
import time
import random
from pprint import pp
from grammar import *

VERBOSE = True

def bottom_up(global_bound, grammar, exs):
    """
    global_bound: int. an upper bound on the size of expression
    exs: list of tuples of environment (the input) and desired output, such as [({'x': 5}, 6), ({'x': 1}, 2)]
    returns: either None if no program can be found that satisfies the input outputs, or the smallest such program. If a program `p` is returned, it should satisfy `all( p.eval(input) == output for input, output in exs )`
    """
    target_outputs = tuple(y for _, y in exs)

    for expr in bottom_up_generator(global_bound, grammar, exs):
        outputs = tuple(expr.eval(input) for input, _ in exs)
        if outputs == target_outputs:
            return expr

    return None

def bottom_up_generator(global_bound, grammar, exs):
    """
    global_bound: int. an upper bound on the size of expression
    exs: list of tuples of environment (the input) and desired output, such as [({'x': 5}, 6), ({'x': 1}, 2)]
    yields: sequence of programs, ordered by expression size, which are semantically distinct on the input examples
    """
    trees = dict() # mapping (type, size) => all values that can be computed of type using an expression of size
    seen = set() # Store seen outputs (observational equivalence filtering)

    def pad(x):
        return (x, type(x))

    def add_tree(key, tree):
        """
        Add a tree to trees[k] if not observationally equivalent to a pre-existing program.

        Use `pad` to store (output, type) pairs to address an issue with how Python thinks True == 1
        """
        out = tuple(pad(tree.eval(x)) for x, _ in exs)
        if out not in seen:
            if key not in trees: 
                trees[key] = [tree]
            else: 
                trees[key].append(tree)    
            seen.add(out)

    # Add all terminals to `trees` as trees of size 1
    for x in grammar.consts:
        yield x
        add_tree((x.return_type, 1), x)

    # Yield a generator for each size?

    # Generate all trees of size `size` and add to `trees`
    for size in range(1, global_bound+1):
        if VERBOSE: print(f"Generating trees of size {size}... ", end='')
        # For each operator, determine how many args it needs and fill in using integer_partitions
        for op in grammar.ops:
            in_types = op.argument_types
            n_inputs = len(in_types)
            for partition in integer_partitions(size - 1 - n_inputs, n_inputs):
                for args in itertools.product(*(trees.get((typ, size+1), []) 
                                                for typ, size in zip(in_types, partition))):
                    tree = op(*args)
                    yield tree
                    add_tree((op.return_type, size), tree)
        
        # print(f"size {size} keys:")
        # for k in trees.keys():
        #     print(k)

        if VERBOSE: print(f"|trees| = {sum(len(l) for l in trees.values())}")

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
