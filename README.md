# arc

## Sanity check
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
