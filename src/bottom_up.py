import itertools
import time
import random
import util
from pprint import pp
from grammar import *

VERBOSE = False

def gen_zb():
    return [bool(random.randint(0,1)) for _ in range(Z_SIZE)]

def gen_zn():
    return [random.randint(Z_LO, Z_HI) for _ in range(Z_SIZE)]

def add_zs(exs):
    """
    Generate z_i = z_i_b, z_i_n for each x_i and add to env
    """
    for x, y in exs:
        if 'z_n' not in x:
            x['z_n'] = gen_zn()
        if 'z_b' not in x:
            x['z_b'] = gen_zb()

def bottom_up(global_bound, operators, constants, exs):
    """
    global_bound: int. an upper bound on the size of expression
    operators: list of classes, such as [Times, If, ...]
    constants: list of possible leaves in syntax tree, such as [Number(1)]. Variables can also be leaves, but these are automatically inferred from `exs`
    exs: list of tuples of environment (the input) and desired output, such as [({'x': 5}, 6), ({'x': 1}, 2)]
    returns: either None if no program can be found that satisfies the input outputs, or the smallest such program. If a program `p` is returned, it should satisfy `all( p.eval(input) == output for input, output in exs )`
    """
    target_outputs = tuple(y for _, y in exs)

    # generate z_i = z_i_b, z_i_n for each x_i and add to env
    add_zs(exs)

    for expr in bottom_up_generator(global_bound, operators, constants, exs):
        outputs = tuple(expr.eval(input) for input, _ in exs)
        if outputs == target_outputs:
            return expr

    return None

def bottom_up_generator(global_bound, operators, constants, exs):
    """
    global_bound: int. an upper bound on the size of expression
    operators: list of classes, such as [Times, If, ...]
    constants: list of possible leaves in syntax tree, such as [Number(1)]. Variables can also be leaves, but these are automatically inferred from `exs`
    exs: list of tuples of environment (the input) and desired output, such as [({'x': 5}, 6), ({'x': 1}, 2)]
    yields: sequence of programs, ordered by expression size, which are semantically distinct on the input examples
    """
    # variables: use z_b[0] .. z_b[n-1] and z_n[0] .. z_n[n-1]
    zb = [Zb(Num(i)) for i in range(Z_SIZE)]
    zn = [Zn(Num(i)) for i in range(Z_SIZE)]

    vars_and_consts = constants + zb + zn

    trees = dict() # mapping (type, size) => all values that can be computed of type using an expression of size
    seen = set() # Store seen outputs (observational equivalence filtering)

    def add_tree(key, tree):
        """Add a tree to trees[k] if not observationally equivalent to a pre-existing program"""
        out = tuple(tree.eval(x) for x, _ in exs)
        if out not in seen:
            if key not in trees: 
                trees[key] = [tree]
            else: 
                trees[key].append(tree)    
            seen.add(out)

    # Add all terminals to `trees` as trees of size 1
    for x in vars_and_consts:
        yield x
        add_tree((x.return_type, 1), x)

    # Generate all trees of size `size` and add to `trees`
    for size in range(1, global_bound+1):
        if VERBOSE: print(f"Generating trees of size {size}... ", end='')
        # For each operator, determine how many args it needs and fill in using integer_partitions
        for op in operators:
            in_types = op.argument_types
            n_inputs = len(in_types)
            assert n_inputs != 0, "operators shouldn't be terminals"
            for partition in integer_partitions(size - 1 - n_inputs, n_inputs):
                for args in itertools.product(*(trees.get((t, s+1), []) 
                                              for t, s in zip(in_types, partition))):
                    tree = op(*args)
                    # print(tree)
                    yield tree
                    add_tree((op.return_type, size), tree)
        # print(f"keys: {keys}")
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
    operators = [Program, Rect, Point]
    terminals = [# Num(i) for i in range(max(IMG_WIDTH, IMG_HEIGHT))
                 ]

    # collection of input-output specifications
    test_cases = [
        # [({}, 1)],
        # [({}, Point(Num(1), Num(1)).eval({}))],
        # [({}, Rect(Point(Num(1), Num(1)), 
        #            Point(Num(5), Num(6))).eval({}))],
        [({"z_n": [3, 3, 5, 5, 0, 0]}, 
          Rect(Point(Num(3), Num(3)), 
               Point(Num(5), Num(5))).eval({}))],
        [({"z_n": [0, 1, 2, 3, 4, 5]}, 
          Program(Rect(Point(Num(1), Num(1)), 
                       Point(Num(2), Num(2))),
                  Rect(Point(Num(4), Num(4)), 
                       Point(Num(5), Num(5)))).eval({})),
         # ({"z_n": [1, 2, 3, 4, 5, 6]}, 
         #  Program(Rect(Point(Num(2), Num(2)), 
         #               Point(Num(3), Num(3))),
         #          Rect(Point(Num(5), Num(5)), 
         #               Point(Num(6), Num(6)))).eval({}))
         ],
        [({"z_n": [0, 0, 1, 2, 4, 5]}, 
          Program(Rect(Point(Num(1), Num(1)), 
                       Point(Num(2), Num(2))),
                  Rect(Point(Num(4), Num(4)), 
                       Point(Num(5), Num(5)))).eval({})),
         ({"z_n": [0, 0, 2, 2, 5, 5]}, 
          Program(Rect(Point(Num(2), Num(2)), 
                       Point(Num(2), Num(2))),
                  Rect(Point(Num(5), Num(5)), 
                       Point(Num(5), Num(5)))).eval({})),],
        # [({"z_n": list(range(Z_SIZE))}, 
        #   Rect(Point(1,1), Point(5,6)))],
        # [({"z_n": [100+x for x in range(Z_SIZE)]}, 
        #   ((100,100), (105,106)))],
    ]
    bound = 17
    for test_case in test_cases:
        start_time = time.time()
        expr = bottom_up(bound, operators, terminals, test_case)
        print(f"synthesized program:\t {expr.pretty_print() if expr is not None else 'None'} in {time.time() - start_time} seconds")

    # print(" [+] bottom-up synthesis passes tests")

if __name__ == "__main__":
    test_bottom_up()
