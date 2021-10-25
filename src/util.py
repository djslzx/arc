def avg(it):
    s = 0
    n = 0
    for x in it:
        s += x
        n += 1
    return s/n

def sum_sq(it):
    return sum(x*x for x in it)
