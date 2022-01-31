import util
from gen import *

if __name__ == '__main__':
    n = 10
    data = util.load('../data/full-exs.dat')
    for i in range(n):
        bmps, p = next(data)
        prog = deserialize(p)
        print(prog, bmps)
