import torch as T
import random as R 
import itertools as it
from typing import Callable, List, Dict, Tuple, Optional
from collections import namedtuple

import util
import viz

def dim(pts):
    if not pts: return 0, 0
    xs, ys = util.unzip(pts)
    return (max(xs) - min(xs) + 1,
            max(ys) - min(ys) + 1)

def in_bounds(pts, w, h):
    width, height = dim(pts)
    return width <= w and height <= h

def shift(pts, x0, y0):
    """
    Shift pts so that min_x {pts} = x0, min_y {pts} = y0
    """
    if not pts: return pts
    xs, ys = util.unzip(pts)
    min_x = min(xs)
    min_y = min(ys)
    if min_x < x0:
        xs = [x + (x0 - min_x) for x in xs]
    if min_y < y0:
        ys = [y + (y0 - min_y) for y in ys]
    return list(zip(xs, ys))


State = namedtuple('State', 'x y prev_dx prev_dy pts')

def orthogonal_ant(w, h):
    def step(s: State):
        dx, dy = 0, 0
        if R.randint(0, 1):
            dx = R.choice([-1, 1])
        else:
            dy = R.choice([-1, 1])
        return dx, dy
    return random_walk(w, h, step)

def diagonal_ant(w, h):
    def step(s: State):
        return R.choice([
            (dx, dy) for (dx, dy) in it.product([-1, 0, 1], [-1, 0, 1])
            if not (dx == 0 and dy == 0) and not (s.prev_dx == -dx and s.prev_dy == -dy)
        ])
    return random_walk(w, h, step)

def non_repeating_ant(w, h):
    def step(s: State):
        options = [
            (dx, dy) for (dx, dy) in it.product([-1, 0, 1], [-1, 0, 1])
            if s.x + dx < w and s.y + dy < h and not (s.x + dx, s.y + dy) in s.pts
        ]
        return R.choice(options) if options else None
    return random_walk(w, h, step)

def random_walk(w, h, step: Callable[[State], Optional[Tuple[int, int]]]):
    x, y = 0, 0
    pts = []
    prev_dx, prev_dy = 0, 0
    while in_bounds(pts + [(x, y)], w, h):
        pts.append((x, y))
        s = step(State(x, y, prev_dx, prev_dy, pts))
        if s is None: break
        dx, dy = s
        prev_dx, prev_dy = dx, dy
        x += dx
        y += dy
    return shift(pts, 0, 0)

def random_colors(w, h):
    return (T.rand(w, h) * 10).long()

def make_sprite(w, h, W, H):
    # ensure that generated pts do not define a point, line, or rect
    assert w > 1 and h > 1, f"Sprites of width/height 1 are either points or lines"
    ant = non_repeating_ant
    while classify(pts := ant(w, h)) != 'Sprite': pass
    return util.make_bitmap(lambda p: p in pts, W, H)

def make_multicolored_sprite(w, h, W, H):
    sprite = make_sprite(w, h, W, H)
    colors = random_colors(w, h)
    return sprite * util.pad_mat(colors, W, H)

def connected(pts):
    """
    Checks whether a sequence of points represents a connected subgrid (diagonal connections OK)
    """
    for i in [0, 1]:
        s = sorted(pts, key=lambda p: p[i])
        for j in range(1, len(s)):
            if abs(s[j][i] - s[j-1][i]) > 1:
                return False
    return True

def is_rect(pts):
    if not connected(pts): return False
    xs, ys = util.unzip(pts)
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    return all((x,y) in pts
                for x in range(min_x, max_x+1)
               for y in range(min_y, max_y+1))

def is_line(pts):
    if not connected(pts): return False
    xs, ys = util.unzip(pts)
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    return min_x == max_x \
        or min_y == max_y \
        or all(y == min_y + (x - min_x) for x,y in pts)

def is_point(pts):
    xs, ys = util.unzip(pts)
    return min(xs) == max(xs) and min(ys) == max(ys) 

def classify(pts):
    if not pts: return 'Empty'
    if is_point(pts): return 'Point'
    if is_line(pts): return 'Line'
    if is_rect(pts): return 'Rect'
    return 'Sprite'

def test_connected():
    tests = [
        ([], True),
        (["____",
          "____",
          "_#__",
          "____"], True),
        (["_#__",
          "____",
          "_#__",
          "____"], False),
        (["__#__",
          "_#__",
          "###_",
          "____"], True),
        (["_#__",
          "_##_",
          "____",
          "__##"], False),
    ]
    for img, ans in tests:
        t = util.img_to_tensor(img, w=W, h=H)
        pts = util.tensor_to_pts(t)
        o = connected(pts)
        assert o == ans, \
            f"Classified {t} as {o}, expected {ans}"
    print("[+] passed test_connected")

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
          "____"], 'Sprite'),
        (["_#__",
          "_#__",
          "_##_",
          "____"], 'Sprite'),
        (["_#__",
          "_##_",
          "_#__",
          "____"], 'Sprite'),
    ]
    for img, ans in tests:
        t = util.img_to_tensor(img, w=W, h=H)
        pts = util.tensor_to_pts(t)
        o = classify(pts)
        assert o == ans, \
            f"Classified {t} as {o}, expected {ans}"
    print("[+] passed test_classify")


if __name__ == '__main__':
    W, H = 8, 8
    w, h = 4, 4
    test_connected()
    test_classify(W, H)
    mat_w, mat_h = 3, 3
    for n in range(2, 8):
        sprites = T.stack([T.stack([make_multicolored_sprite(n, n, 16, 16)
                                    for _ in range(mat_h)])
                           for _ in range(mat_w)])
        viz.viz_grid(sprites)
