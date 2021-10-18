# arc

## Sanity check
### TODO
- [X] enforce invariants about rectangles 
  - can't enforce with `assert`s in generation stage as this will halt the program, but this needs to be addressed before reaching the learning stage, or we'll end up with invalid candidate solutions
  - enforced via `satisfies_invariants` fn
- [X] swap between optimizing `f` and optimizing `z`
- [ ] think about witness functions
- [ ] think about probabilistic programs

### Grammar
Given `z = (z_b, z_n)`, a tuple of random vectors over booleans and integers;

```
p := rect | rect p
rect := (pt, pt)
pt := (n, n)
n := 0 | 1 | ... | 9 | n + n | n * n | n - n | n / n
   | if b then n else n | z_n[i]
b := z_b[i] | false | not b

```

### Test task
Given an 8x8 pixel image `x` containing two rectangles (one 2x2 rect and one larger rect straddling a corner), generate a program `f` that maps a random input `z` to `x`.

#### Training data
x1:
```
________
________
_____##_
_____##_
________
________
#####___
#####___
```

x2:
```
###_____
###_____
###___##
###___##
________
________
________
________
```

### Strategy
Use inductive synthesis (bottom-up enumeration with decision trees) to find `f'` given `x_i` and `z_i` such that `f'(z_i) ~ x_i` for all `x_i`.

Once an `f'` is found that approximates each `x_i` (minimizing wrt L2 norm (sum of squared diffs)), fix `f'` and perturb random inputs `z_i` giving `z_i'` so that `f'(z')` better approximates `x_i`.

### Notes
#### Keeping local optima across iterations
It seems that carrying through optimal solutions as candidates in the next phase of the search (using prior Z values as candidates for future Z's, or prior f's for future f's) hampers the search from considering other solutions.

#### Grammar size
It looks like using a larger grammar (i.e. w/ arithmetic expressions) leads the search away from the optimal solution for simple expressions that don't require arithmetic expressions, e.g. 
```
[
  ({}, Rect(Point(Num(0), Num(0)), Point(Num(1), Num(1)))),
  ({}, Rect(Point(Num(1), Num(1)), Point(Num(2), Num(2)))),
],
```
The search space still includes the optimal solutions, because we're augmenting the grammar. So it seems that there's something about the way we conduct the search that prevents these optimal, shorter solutions from being considered over longer and less optimal solutions.  Maybe longer solutions perform better within each round, so we never hold onto a shorter solution long enough to optimize Z to fit it.

#### Increasing samples of Z
Increasing the number of random samples in the `opt_z` step seems to help, but we still get overcomplicated expressions to which we fit Z:
```
Synthesized program:	 ((z_n[3], z_n[3]), (z_n[5], (- z_n[0] z_n[4]))), 
Z: [([3, 1, 2, 0, 2, 1], [True, True, True, True, True, True]), ([3, 3, 3, 1, 1, 2], [True, True, False, False, False, True])] in 1.370865821838379 seconds
```

#### Storing/managing `z`
- associate one `z_i` with each `x_i` and learn a pattern for manipulating `z_i` to get `x_i`
- generate each `z_i` as part of the problem input?

#### False and Num(0)
Python treats `bool` as a subclass of `int`, so 
```
False == 0
>> True
```

This means that generated programs will conflate `False` and `0`. (We don't need to worry about `True` because we don't have a `True` literal.)

