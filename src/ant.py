import torch as T
import random as R 
import util
from viz import viz

def make_bitmap(f, W, H):
    return T.tensor([[f((x, y))
                      for x in range(W)]
                     for y in range(H)]).float()

def unzip(l):
    return tuple(list(x) for x in zip(*l))

def dim(pts):
    if not pts: return 0, 0
    xs, ys = unzip(pts)
    return (max(xs) - min(xs) + 1,
            max(ys) - min(ys) + 1)

def in_bounds(pts, w, h):
    width, height = dim(pts)
    return width <= w and height <= h

def shift(pts, x0, y0):
    xs, ys = unzip(pts)
    min_x = min(xs)
    min_y = min(ys)
    if min_x < x0:
        xs = [x + (x0 - min_x) for x in xs]
    if min_y < y0:
        ys = [y + (y0 - min_y) for y in ys]
    return list(zip(xs, ys))

def ant(x0, y0, w, h, W, H):
    x, y = 0, 0
    pts = []
    while in_bounds(pts + [(x,y)], w, h):
        pts.append((x, y))
        dx = R.randint(-1, 1)
        dy = R.randint(-1, 1) if dx != 0 else R.choice([-1, 1]) # makes dy depend on dx :(
        x += dx
        y += dy
    pts = shift(pts, x0, y0)
    return make_bitmap(lambda p: p in pts, W, H)

def as_colored_pts(bmp):
    H, W = bmp.shape
    return [(x,y,c) 
            for y in range(H)
            for x in range(W)
            if (c := bmp[y][x]) > 0]

def check(pred, bmp, color):
    H, W = bmp.shape
    return all(bmp[y][x] == (color if pred((x, y)) else 0)
               for y in range(H)
               for x in range(W))

def is_rect(bmp):
    pts = as_colored_pts(bmp)
    if not pts: return False
    xs, ys, cs = unzip(pts)
    if not all(c == cs[0] for c in cs[1:]): return False
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    return (bmp[min_y:max_y+1, 
                min_x:max_x+1] > 0).all()

def is_line(bmp):
    pts = as_colored_pts(bmp)
    if not pts: return False
    xs, ys, cs = unzip(pts)
    if not all(c == cs[0] for c in cs[1:]): return False
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)

    # vertical
    if min_x == max_x:
        return check(lambda p: p[0] == min_x and min_y <= p[1] <= max_y, bmp, cs[0])
    # horizontal
    if min_y == max_y:
        return check(lambda p: p[1] == min_y and min_x <= p[0] <= max_x, bmp, cs[0])
    # diagonal
    return check(lambda p: (min_x <= p[0] <= max_x and
                            min_y <= p[1] <= max_y and
                            p[1] == min_y + (p[0] - min_x)), bmp, cs[0])

def is_point(bmp):
    pts = as_colored_pts(bmp)
    if not pts: return False
    xs, ys, cs = unzip(pts)
    if not all(c == cs[0] for c in cs[1:]): return False
    return min(xs) == max(xs) and min(ys) == max(ys) 

def classify(bmp):
    if is_point(bmp): return 'Point'
    if is_line(bmp): return 'Line'
    if is_rect(bmp): return 'Rect'
    return 'Shape'

def test_classify(W, H):
    tests = [
        (["____",
          "____",
          "_#__",
          "____"], 'Point'),
        (["____",
          "____",
          "_#__",
          "_#__"], 'Line'),
        (["____",
          "____",
          "_###",
          "____"], 'Line'),
        (["_#__",
          "__#_",
          "___#",
          "____"], 'Line'),
        (["_##_",
          "_##_",
          "_##_",
          "____"], 'Rect'),
        (["_##_",
          "_#__",
          "_##_",
          "____"], 'Shape'),
    ]
    for q, a in tests:
        q = util.img_to_tensor(q, w=W, h=H)
        o = classify(q)
        assert o == a, \
            f"Classified {q} as {o}, expected {a}"
    print("[+] passed test_classify")

if __name__ == '__main__':
    test_classify(W=8, H=8)
    # for i in range(B_W):
    #     bmp = ant(i,i,3,3)
    #     viz(bmp, title=classify(bmp))
