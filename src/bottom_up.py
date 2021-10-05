import itertools
import time
from grammar import *

def bottom_up(global_bound, operators, constants, input_outputs):
    """
    global_bound: int. an upper bound on the size of expression
    operators: list of classes, such as [Times, If, ...]
    constants: list of possible leaves in syntax tree, such as [Number(1)]. Variables can also be leaves, but these are automatically inferred from `input_outputs`
    input_outputs: list of tuples of environment (the input) and desired output, such as [({'x': 5}, 6), ({'x': 1}, 2)]
    returns: either None if no program can be found that satisfies the input outputs, or the smallest such program. If a program `p` is returned, it should satisfy `all( p.evaluate(input) == output for input, output in input_outputs )`
    """
    target_outputs = tuple( y for _, y in input_outputs )

    for expression in bottom_up_generator(global_bound, operators, constants, input_outputs):
        outputs = tuple(expression.evaluate(input) for input, _ in input_outputs)
        if outputs == target_outputs:
            return expression

    return None

def bottom_up_generator(global_bound, operators, constants, input_outputs):
    """
    global_bound: int. an upper bound on the size of expression
    operators: list of classes, such as [Times, If, ...]
    constants: list of possible leaves in syntax tree, such as [Number(1)]. Variables can also be leaves, but these are automatically inferred from `input_outputs`
    input_outputs: list of tuples of environment (the input) and desired output, such as [({'x': 5}, 6), ({'x': 1}, 2)]
    yields: sequence of programs, ordered by expression size, which are semantically distinct on the input examples
    """

    def make_variable(variable_name, variable_value):
        """
        make variables that wraps the variable name in the correct class depending on the type of the variable
        """
        from part4 import StringVariable
        if isinstance(variable_value, int): return NumberVariable(variable_name)
        if isinstance(variable_value, str): return StringVariable(variable_name)
        assert False, "only numbers and strings are supported as variable inputs"

    variables = list({make_variable(variable_name, variable_value)
                      for inputs, _ in input_outputs
                      for variable_name, variable_value in inputs.items() })
    variables_and_constants = constants + variables

    # mapping (type, size) => all values that can be computed of type using an expression of size
    trees = dict()
    # Store seen outputs (observational equivalence filtering)
    seen = set()

    def add_tree(key, tree):
        """Add a tree to trees[k] if not observationally equivalent to a pre-existing program"""
        # print(f"add_tree: trees[k={key}]={tree}, input_outputs={input_outputs}")
        out = tuple(tree.evaluate(inputs) for inputs, _ in input_outputs)
        if out not in seen:
            if key not in trees: 
                trees[key] = [tree]
            else: 
                trees[key].append(tree)    
            seen.add(out)

    # Add all terminals to `trees` as trees of size 1
    # print(*(type(v) for v in variables_and_constants))
    for x in variables_and_constants:
        yield x
        add_tree((x.return_type, 1), x)

    # Generate all trees of size `size` and add to `trees`
    for size in range(1, global_bound+1):
        # print(f"Generating trees of size {size}... ", end='')
        # For each operator, determine how many args it needs and fill in using integer_partitions
        for op in operators:
            in_types = op.argument_types
            n_inputs = len(in_types)
            assert n_inputs != 0, "operators shouldn't be terminals"
            for partition in integer_partitions(size - 1 - n_inputs, n_inputs):
                for args in itertools.product(*(trees.get((t, s+1), []) 
                                                for t, s in zip(in_types, partition))):
                    tree = op(*args)
                    yield tree
                    add_tree((op.return_type, size), tree)
        # print(f"|trees| = {sum(len(l) for l in trees.values())}")

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
    operators = [Plus,Times,LessThan,And,Not,If]
    terminals = [FALSE(),Number(0),Number(1),Number(-1)]

    # collection of input-output specifications
    test_cases = []
    test_cases.append([({"x": 1}, 1),
                       ({"x": 4}, 16),
                       ({"x": 5}, 25)])
    test_cases.append([({"x": 1, "y": 2}, 1),
                       ({"x": 5, "y": 2}, 2),
                       ({"x": 99, "y": 98}, 98),
                       ({"x": 97, "y": 98}, 97),])
    test_cases.append([({'x':10, 'y':7}, 17),
	                    ({'x':4, 'y':7}, -7),
	                    ({'x':10, 'y':3}, 13),
	                    ({'x':1, 'y':-7}, -6),
	                    ({'x':1, 'y':8}, -8)])
    test_cases.append([({'x':10, 'y':10}, 20),
                       ({'x':15, 'y':15}, 30),
	               ({'x':4, 'y':7}, 16),
	               ({'x':10, 'y':3}, 9),
	               ({'x':1, 'y':-7}, 49),
	               ({'x':1, 'y':8}, 1)])

    # the optimal size of each program that solves the corresponding test case
    optimal_sizes = [3, 6, 10, 17]

    for test_case, optimal_size in zip(test_cases, optimal_sizes):
        assert bottom_up(optimal_size - 1, operators, terminals, test_case) is None, f"you should not be able to solve this test case w/ a program whose syntax tree is of size {optimal_size-1}. the specific test case is {test_case}"

        start_time = time.time()
        expression = bottom_up(optimal_size, operators, terminals, test_case)
        assert expression is not None, f"failed to synthesize a program when the size bound was {optimal_size}. the specific test case is {test_case}"

        print(f"synthesized program:\t {expression.pretty_print()} in {time.time() - start_time} seconds")
        for xs, y in test_case:
            assert expression.evaluate(xs) == y, f"synthesized program {expression.pretty_print()} does not satisfy the following test case: {xs} --> {y}"
            print(f"passes test case {xs} --> {y}")
        print()
    print(" [+] bottom-up synthesis passes tests")

if __name__ == "__main__":
    test_bottom_up()
