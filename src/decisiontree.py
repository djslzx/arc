from grammar import *
from bottom_up import *

import math
import time

global_bound = 10               # FIXME

def covered(cover, n_pts):
    """
    Check whether, for each input-output example `pt`, there is a term covering `pt`
    """
    # TODO: convert to matrix ops?
    covers = cover.values()
    return all(any(c[i] for c in covers)
               for i in range(n_pts))

def dt_works(dt, pts):
    """Returns true if the decision tree works, false otherwise"""
    if dt is None: return False
    return all(dt.eval(x) == y for (x,y) in pts)

def dcsolve(ops, consts, pts):
    """
    ops: list of classes, such as [Times, Not, ...]. Note that `If` does not have to be here, because the decision tree learner inserts such exprs
    consts: list of possible leaves in syntax tree, such as [Number(1)]. Variables can also be leaves, but these are automatically inferred from `pts`
    pts: list of tuples of environment (the input) and desired output, such as [({'x': 5}, 6), ({'x': 1}, 2)]
    returns: an expr `p` which should satisfy `all( p.eval(input) == output for input, output in pts )`
    """
    terms = []              # list(expr)
    preds = set()           # set(expr)
    cover = dict()          # expr -> tuple(bool) : term -> i -> bool
    n_pts = len(pts)

    # add z_b, z_n if not already populated
    add_zs(pts)

    # determine range(f)
    (x,y) = next(iter(pts))
    return_type = type(y).__name__

    # generators
    pred_gen = distinct_pred_gen(global_bound, ops, consts, pts, preds)
    term_gen = distinct_term_gen(global_bound, ops, consts, pts, terms, cover, return_type)

    # Term solver
    while not covered(cover, n_pts):
        terms.append(next(term_gen))

    # Unifier
    dt = learn_decision_tree(cover, terms, preds, range(n_pts))
    while not dt_works(dt, pts):
        next_term = next(term_gen, None)
        if next_term is not None:
            terms.append(next_term)
        preds.add(next(pred_gen))
        dt = learn_decision_tree(cover, terms, preds, range(n_pts))

    # dt should work now
    return dt

def distinct_pred_gen(global_bound, ops, consts, pts, preds):
    """Generates the next distinct predicate"""
    gen = bottom_up_generator(global_bound, ops, consts, pts)
    for t in gen:
        if t.return_type == "bool": # and outs not in seen:
            outs = tuple(t.eval(x) for (x,_) in pts)
            yield (t, outs)

def distinct_term_gen(global_bound, ops, consts, pts, terms, cover, return_type):
    """Generates the next distinct term"""
    gen = bottom_up_generator(global_bound, ops, consts, pts)
    seen = set(cover[s] for s in terms)
    for t in gen:
        cover[t] = tuple(t.eval(x) == y for (x,y) in pts)
        if t.return_type == return_type and cover[t] not in seen:
            yield t
            seen.add(cover[t])
    
def learn_decision_tree(cover, terms, predicates, examples_we_care_about):
    """
    You may find this utility function helpful
    cover: dictionary mapping from expr to tuple of bool's. `cover[e][i] == True` iff expr `e` predicts the correct output for `i`th input
    terms: set of exprs that we can use as leaves in the decision tree
    predicates: predicates we can use as branches in the decision tree. each element of `predicates` should be a tuple of `(expr, outputs)` where `outputs` is a tuple of bool's. Should satisfy `outputs[i] == expr.eval(pts[i][0])`
    examples_we_care_about: a set of integers, telling which input outputs we care about solving for. For example if we are done, then this will be the empty set. If we are just starting out synthesizing the decision tree, then this will be the numbers 0-(len(pts)-1)
    """
    for expr in terms:
        if all( cover[expr][i] for i in examples_we_care_about ):
            return expr

    if len(predicates) == 0: return None # no more predicates to split on

    def information_gain(predicate_info):
        """actually returns -information gain up to a constant ($G$ in paper)"""
        predicate, predicate_outputs = predicate_info

        examples_yes = {i for i in examples_we_care_about if predicate_outputs[i] }
        examples_no = {i for i in examples_we_care_about if not predicate_outputs[i] }
        
        probability_yes = len(examples_yes)/len(examples_we_care_about)
        probability_no = len(examples_no)/len(examples_we_care_about)
        
        entropy_yes = entropy(examples_yes)
        entropy_no = entropy(examples_no)

        return probability_yes * entropy_yes + probability_no * entropy_no

    def entropy(example_indices):
        # entries proportional probability that the term used during evaluation is a specific term
        # len of `distribution` will be the number of terms
        distribution = []

        for expr in terms:
            # calculate probability that we used this expr, assuming uniform distribution over which example is being run
            ps = []
            for example_index in example_indices:
                if not cover[expr][example_index]: # we can't explain this example, definitely are not a candidate term
                    p = 0
                else:
                    p = sum( cover[expr][i] for i in example_indices )
                    p /= sum( cover[other_expr][i]
                              for other_expr in terms
                              if cover[other_expr][example_index]
                              for i in example_indices)
                ps.append(p)
            
            distribution.append(sum(ps))

        # original paper has 1/|pts| term, but we can absorb this into normalizing constant
        z = sum(distribution) # normalizing constant

        return -sum( p/z * math.log(p/z) for p in distribution if p > 0)
        
    predicate, predicate_outputs = min(predicates, key=information_gain)

    left_hand_side_examples = {i for i in examples_we_care_about if predicate_outputs[i]}
    right_hand_side_examples = {i for i in examples_we_care_about if not predicate_outputs[i]}

    predicates = predicates - {(predicate, predicate_outputs)}

    lhs = learn_decision_tree(cover, terms, predicates, left_hand_side_examples)
    if lhs is None: return None
    
    rhs = learn_decision_tree(cover, terms, predicates, right_hand_side_examples)
    if rhs is None: return None

    return If(predicate, lhs, rhs)
    
def test_dcsolve():
    operators = [Plus,Minus,Times,Div,Lt,And,Not,If,Point,Rect]
    terminals = [FALSE()] + [Num(i) for i in range(Z_LO, Z_HI+1)]

    # collection of input-output specifications
    test_cases = [
        [({}, 1)],
        [({}, 10)],
        [({}, Point(1,1))],
        [({}, Rect(Point(1,1), 
                   Point(5,6)))],
        [({"z_n": list(range(Z_SIZE))}, Rect(Point(1,1), 
                                             Point(5,6)))],
        [({"z_n": [100+x for x in range(Z_SIZE)]}, 
          Rect(Point(100,100), Point(105,106)))],
    ]

    for test_case in test_cases:
        start_time = time.time()
        print(f"Testing case: {test_case}")
        expr = dcsolve(operators, terminals, test_case)
        print(f"synthesized program:\t {expr.pretty_print()} in {time.time() - start_time} seconds")
        for xs, y in test_case:
            assert expr.eval(xs) == y, f"synthesized program {expr.pretty_print()} does not satisfy the following test case: {xs} --> {y}"
            print(f"passes test case {xs} --> {y}")
        print()
    print(" [+] dcsolver passes tests")

def test_covered():
    tests = [
        ({0: [True, True, True]}, 3, True),
        ({0: [True, False], 
          1: [False, True]}, 2, True),
        ({0: [True, False, False], 
          1: [False, True, False],
          2: [False, False, True]}, 3, True),
        ({0: [False]}, 1, False),
        ({0: [True, False]}, 2, False),
        ({0: [True, False, False],
          1: [False, True, False]}, 3, False),
    ]
    for a,b,c in tests:
        out = covered(a,b)
        assert out == c, \
            f"Failed test: in={(a,b)}, expected={c}, actual={out}"
    print("covered passes tests")

if __name__ == "__main__":
    # test_covered()
    test_dcsolve()
