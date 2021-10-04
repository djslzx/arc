# arc

## Sanity check
### Grammar
Given `z = (z_b, z_n)`, a tuple of random vectors over booleans and integers;

```
p := rect | rect p
rect := (pt, pt)
pt := (n, n)
n := int | n + n | n * n | n - n | n / n
   | if b then n else n
b := z_b[i]

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
