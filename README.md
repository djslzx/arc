# arc

## Strategy
Use inductive synthesis (bottom-up enumeration with decision trees) to find `f'` given `x_i` and `z_i` such that `f'(z_i) ~ x_i` for all `x_i`.

Once an `f'` is found that approximates each `x_i` (minimizing wrt L2 norm (sum of squared diffs)), fix `f'` and perturb random inputs `z_i` giving `z_i'` so that `f'(z')` better approximates `x_i`.

## Notes
### Measuring instance complexity / bounding search depth for f
We need some principled way of trading off between optimizing f and optimizing z, instead of treating f optimization rounds as a hyperparameter.

One way of doing this might be to somehow measure the complexity in a bitmap and using this to bound the search depth.

Alternatively, we can think of a way of iteratively increasing depth while optimizing z -- maybe optimize z at each depth level? Is there some way we can use dynamic programming to avoid redundancy here?


### Better error functions
Currently, the distance metric treats Rect(0,0,1,1) and Rect(3,3,4,4) as the same distance from Rect(1,1,2,2), although the first expression is a better approximation of the target. As another example, consider the following:

Target
```
#___
#___
____
____
```

A
```
##__
##__
____
____
```

B
```
____
____
____
___#
```

If we only count pixel differences, B is closer to Target than A even though in the grammar, A is only one edit away from Target.

It seems that using some notion of edit distance on the grammar would work better here, but this would require having access to a ground truth expression, which we don't have.

### Keeping local optima across iterations
It seems that carrying through optimal solutions as candidates in the next phase of the search (using prior Z values as candidates for future Z's, or prior f's for future f's) hampers the search from considering other solutions.

### Grammar size
It looks like using a larger grammar (i.e. w/ arithmetic expressions) leads the search away from the optimal solution for simple expressions that don't require arithmetic expressions, e.g. 
```
[
  ({}, Rect(Point(Num(0), Num(0)), Point(Num(1), Num(1)))),
  ({}, Rect(Point(Num(1), Num(1)), Point(Num(2), Num(2)))),
],
```
The search space still includes the optimal solutions, because we're augmenting the grammar. So it seems that there's something about the way we conduct the search that prevents these optimal, shorter solutions from being considered over longer and less optimal solutions.  Maybe longer solutions perform better within each round, so we never hold onto a shorter solution long enough to optimize Z to fit it.

Increasing the number of random samples in the `opt_z` step seems to help, but we still get overcomplicated expressions to which we fit Z:
```
Synthesized program:	 ((z_n[3], z_n[3]), (z_n[5], (- z_n[0] z_n[4]))), 
Z: [([3, 1, 2, 0, 2, 1], [True, True, True, True, True, True]), ([3, 3, 3, 1, 1, 2], [True, True, False, False, False, True])] in 1.370865821838379 seconds
```

### Storing/managing `z`
- associate one `z_i` with each `x_i` and learn a pattern for manipulating `z_i` to get `x_i`
- generate each `z_i` as part of the problem input?

### False and Num(0)
Python treats `bool` as a subclass of `int`, so 
```
False == 0
>> True
```

This means that generated programs will conflate `False` and `0`. (We don't need to worry about `True` because we don't have a `True` literal.)

