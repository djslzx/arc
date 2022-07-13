import pdb
import itertools as it
import torch as T
import torch.nn.functional as F
import random
from typing import List, Tuple, Dict, Callable, Type

import util
import ant

# bitmap size constants
B_W = 16
B_H = 16
SPRITE_MAX_SIZE = 6

LIB_SIZE = 8                    # number of z's, sprites
Z_LO = 0                        # min poss value in z_n
Z_HI = max(B_W, B_H)            # max poss value in z_n
Z_IGNORE = -1                   # ignore z's that have this value
IMG_IGNORE = -1                 # ignore pixels that have this value
FULL_LEXICON = ([i for i in range(Z_LO, Z_HI+1)] +
                [f'z_{i}' for i in range(LIB_SIZE)] +
                [f'S_{i}' for i in range(LIB_SIZE)] +
                ['x_max', 'y_max',
                 '~', '+', '-', '*', '<', '&', '?',
                 'P', 'L', 'CR', 'SR'
                 'H', 'V', 'T', '#', 'o', '@', '!', '{', '}', '(', ')'])
OLD_LEXICON = (
    [i for i in range(Z_LO, Z_HI+1)] +
    [f'z_{i}' for i in range(LIB_SIZE)] +
    [f'S_{i}' for i in range(LIB_SIZE)] +
    ['x_max', 'y_max', 'P', 'L', 'CR', 'SR', '{', '}', '(', ')']
)
SIMPLE_LEXICON = (
    [i for i in range(Z_LO, Z_HI+1)] +
    [f'z_{i}' for i in range(LIB_SIZE)] +
    [f'S_{i}' for i in range(LIB_SIZE)] +
    [f'CS_{i}' for i in range(LIB_SIZE)] +
    ['x_max', 'y_max', 'P', 'CL', 'LL', 'CR', 'SR', '{', '}', '(', ')']
)

class Visited:
    def accept(self, visitor):
        assert False, f"`accept` not implemented for {type(self).__name__}"

class Expr(Visited):
    def accept(self, visitor): assert False, f"not implemented for {type(self).__name__}"
    def __repr__(self): return str(self)
    def __eq__(self, other): return str(self) == str(other)
    def __hash__(self): return hash(str(self))
    def __ne__(self, other): return str(self) != str(other)
    def __gt__(self, other): return str(self) > str(other)
    def __lt__(self, other): return str(self) < str(other)
    def eval(self, env={}, height=B_H, width=B_W):
        return self.accept(Eval(env, height, width))
    def extract_indices(self, type):
        def f_map(t, *args):
            if t == type:
                assert len(args) == 1, f"Visited {type} but got an unexpected number of args"
                i = args[0]
                return [i]
            else:
                return []
        def f_reduce(type, *children):
            return util.uniq([x for child in children for x in child])
        return self.accept(MapReduce(f_reduce, f_map))
    def zs(self):
        return self.extract_indices(Z)
    def sprites(self):
        return self.extract_indices(Sprite)
    def csprites(self):
        return self.extract_indices(ColorSprite)
    def count_leaves(self):
        def f_map(type, *args): return 1
        def f_reduce(type, *children): return sum(children)
        return self.accept(MapReduce(f_reduce, f_map))
    def leaves(self):
        def f_map(type, *args):
            return [[type(*args)]]
        def f_reduce(type, *children):
            return [[type] + path
                    for child in children
                    for path in child]
        return self.accept(MapReduce(f_reduce, f_map))
    def perturb_leaves(self, p, range=(0, 2)):
        n_perturbed = 0
        # range = self.range(envs=[])
        def perturb(expr):
            nonlocal n_perturbed
            if random.random() < p:
                n_perturbed += 1
                return expr.accept(Perturb(range))
            else:
                return expr
        def perturb_leaf(type, *args):
            return perturb(type(*args))
        def perturb_node(type, *children):
            try:
                return perturb(type(*children))
            except UnimplementedError:
                return type(*children)
        return self.accept(MapReduce(f_map=perturb_leaf, f_reduce=perturb_node))
    def lines(self):
        try:
            return self.bmps
        except AttributeError:
            return []
    def add_line(self, line):
        assert isinstance(self, Seq)
        assert type(line) in [Point, CornerLine, LengthLine, CornerRect, SizeRect, Sprite, ColorSprite]
        return Seq(*self.bmps, line)
    def simplify_indices(self):
        zs = self.zs()
        sprites = self.sprites()
        csprites = self.csprites()
        return self.accept(SimplifyIndices(zs, sprites, csprites))
    def serialize(self): return self.accept(Serialize())
    def well_formed(self):
        try:
            return self.accept(WellFormed())
        except (AssertionError, AttributeError):
            return False
    def range(self, envs): return self.accept(Range(envs))
    def size(self):
        """Counts both leaves and non-leaf nodes"""
        def f_map(type, *args): return 1 if type != Sprite else 0
        def f_reduce(type, *children): return 1 + sum(children)
        self.accept(MapReduce(f_reduce, f_map))
    def literal_str(self):
        def to_str(type: Type[Expr], *args):
            arg_str = " ".join([str(arg) for arg in args])
            return f'({type.__name__} {arg_str})'
        return self.accept(MapReduce(to_str, to_str))
    def __len__(self): return self.size()
    def __str__(self):
        return self.accept(Print())

class Grammar:
    def __init__(self, ops, consts):
        self.ops = ops
        self.consts = consts

def seed_zs(lo=Z_LO, hi=Z_HI, n_zs=LIB_SIZE):
    return (T.rand(n_zs) * (hi - lo) - lo).long()

def seed_sprites(n_sprites=LIB_SIZE, height=B_H, width=B_W):
    width_popn = list(range(2, min(width, SPRITE_MAX_SIZE)))
    height_popn = list(range(2, min(height, SPRITE_MAX_SIZE)))
    return T.stack([ant.make_sprite(w=random.choices(population=width_popn,
                                                     weights=[1/(1+w) for w in width_popn],
                                                     k=1)[0],
                                    h=random.choices(population=height_popn,
                                                     weights=[1/(1+h) for h in height_popn],
                                                     k=1)[0],
                                    W=width,
                                    H=height)
                    for _ in range(n_sprites)])

def seed_color_sprites(n_sprites=LIB_SIZE, height=B_H, width=B_W):
    width_popn = list(range(2, min(width, SPRITE_MAX_SIZE)))
    height_popn = list(range(2, min(height, SPRITE_MAX_SIZE)))
    return T.stack([ant.make_multicolored_sprite(
        w=random.choices(population=width_popn, weights=[1/(1+w) for w in width_popn], k=1)[0],
        h=random.choices(population=height_popn, weights=[1/(1+h) for h in height_popn], k=1)[0],
        W=width,
        H=height)
        for _ in range(n_sprites)])

def seed_envs(n_envs):
    # FIXME: add color sprites (make normal sprites then apply colors)
    return [{'z': seed_zs(),
             'sprites': seed_sprites(),
             'color-sprites': seed_color_sprites()
             }
            for _ in range(n_envs)]

# class IllFormedError(Exception): pass
# class IllFormed(Expr):
#     def __init__(self): pass
#     def accept(self, v): raise IllFormedError(f"Visitor {v} tried to visit ill-formed expression.")

class Nil(Expr):
    in_types = []
    out_type = 'bool'
    def __init__(self): pass
    def accept(self, v): return v.visit_Nil()

class Num(Expr):
    in_types = []
    out_type = 'int'
    def __init__(self, n): self.n = n
    def accept(self, v): return v.visit_Num(self.n)

class XMax(Expr):
    in_types = []
    out_type = 'int'
    def __init__(self): pass
    def accept(self, v): return v.visit_XMax()
    
class YMax(Expr):
    in_types = []
    out_type = 'int'
    def __init__(self): pass
    def accept(self, v): return v.visit_YMax()

class Z(Expr):
    in_types = []
    out_type = 'int'
    def __init__(self, i): self.i = i
    def accept(self, v): return v.visit_Z(self.i)

class Not(Expr):
    in_types = ['bool']
    out_type = 'bool'
    def __init__(self, b): self.b = b
    def accept(self, v): return v.visit_Not(self.b)

class Plus(Expr):
    in_types = ['int', 'int']
    out_type = 'int'
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def accept(self, v): return v.visit_Plus(self.x, self.y)

class Minus(Expr):
    in_types = ['int', 'int']
    out_type = 'int'
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def accept(self, v): return v.visit_Minus(self.x, self.y)

class Times(Expr):
    in_types = ['int', 'int']
    out_type = 'int'
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def accept(self, v): return v.visit_Times(self.x, self.y)

class Lt(Expr):
    in_types = ['bool', 'bool']
    out_type = 'bool'
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def accept(self, v): return v.visit_Lt(self.x, self.y)

class And(Expr):
    in_types = ['bool', 'bool']
    out_type = 'bool'
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def accept(self, v): return v.visit_And(self.x, self.y)

class If(Expr):
    in_types = ['bool', 'int', 'int']
    out_type = 'int'
    def __init__(self, b, x, y):
        self.b = b
        self.x = x
        self.y = y
    def accept(self, v): return v.visit_If(self.b, self.x, self.y)

class CornerLine(Expr):
    """
    A corner-to-corner representation of a line. The line is represented by two corners (x_1, y_1) and (x_2, y_2)
    and includes the corner points and the line between them.  Taken together, the two points must form a
    horizontal, vertical, or diagonal line.
    """
    in_types = ['int', 'int', 'int', 'int', 'int']
    out_type = 'bitmap'
    def __init__(self, x1, y1, x2, y2, color=Num(1)):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color
    def accept(self, v): return v.visit_CornerLine(self.x1, self.y1, self.x2, self.y2, self.color)

class LengthLine(Expr):
    """
    A corner-direction-length representation of a line. The line is represented by a point (x, y),
    a direction (dx, dy), and a length l.  The resulting line must be horizontal, vertical, or diagonal.
    """
    in_types = ['int', 'int', 'int', 'int', 'int', 'int']
    out_type = 'bitmap'
    def __init__(self, x, y, dx, dy, l, color=Num(1)):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.l = l
        self.color = color
    def accept(self, v): return v.visit_LengthLine(self.x, self.y, self.dx, self.dy, self.l, self.color)

class Point(Expr):
    in_types = ['int', 'int', 'int']
    out_type = 'bitmap'
    def __init__(self, x, y, color=Num(1)):
        self.x = x
        self.y = y
        self.color = color
    def accept(self, v): return v.visit_Point(self.x, self.y, self.color)

class CornerRect(Expr):
    """
    A rectangle specified by two corners (min and max x- and y-values)
    """
    in_types = ['int', 'int', 'int', 'int', 'int']
    out_type = 'bitmap'
    def __init__(self, x_min, y_min, x_max, y_max, color=Num(1)):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.color = color
    def accept(self, v): return v.visit_CornerRect(self.x_min, self.y_min, self.x_max, self.y_max, self.color)

class SizeRect(Expr):
    """
    A rectangle specified by a corner, a width, and a height.
    """
    in_types = ['int', 'int', 'int', 'int', 'int']
    out_type = 'bitmap'
    def __init__(self, x, y, w, h, color=Num(1)):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
    def accept(self, v): return v.visit_SizeRect(self.x, self.y, self.w, self.h, self.color)

class Sprite(Expr):
    in_types = ['int', 'int', 'int']
    out_type = 'bitmap'
    def __init__(self, i, x=Num(0), y=Num(0), color=Num(1)):
        self.i = i
        self.x = x
        self.y = y
        self.color = color
    def accept(self, v): return v.visit_Sprite(self.i, self.x, self.y, self.color)

class ColorSprite(Expr):
    in_types = ['int', 'int', 'int']
    out_type = 'bitmap'
    def __init__(self, i, x=Num(0), y=Num(0)):
        self.i = i
        self.x = x
        self.y = y
    def accept(self, v): return v.visit_ColorSprite(self.i, self.x, self.y)

class Seq(Expr):
    in_types = ['list(bitmap)']
    out_type = 'bitmap'
    def __init__(self, *bmps):
        self.bmps = bmps
    def accept(self, v): return v.visit_Seq(self.bmps)

class Join(Expr):
    in_types = ['bitmap', 'bitmap']
    out_type = 'bitmap'
    def __init__(self, bmp1, bmp2):
        self.bmp1 = bmp1
        self.bmp2 = bmp2
    def accept(self, v): return v.visit_Join(self.bmp1, self.bmp2)

class Apply(Expr):
    """Applies a transformation to a bitmap"""
    in_types = ['transform', 'bitmap']
    out_type = 'bitmap'
    def __init__(self, f, bmp):
        self.f = f
        self.bmp = bmp
    def accept(self, v): return v.visit_Apply(self.f, self.bmp)

class Repeat(Expr):
    in_types = ['transform', 'int']
    out_type = 'transform'
    def __init__(self, f, n):
        self.f = f
        self.n = n
    def accept(self, v): return v.visit_Repeat(self.f, self.n)

# class Intersect(Expr):
#     in_types = ['bitmap']
#     out_type = 'transform'
#     def __init__(self, bmp):
#         self.bmp = bmp
#     def accept(self, v): return v.visit_Intersect(self.bmp)

class HFlip(Expr):
    in_types = []
    out_type = 'transform'
    def __init__(self): pass
    def accept(self, v): return v.visit_HFlip()

class VFlip(Expr):
    in_types = []
    out_type = 'transform'
    def __init__(self): pass
    def accept(self, v): return v.visit_VFlip()

class Translate(Expr):
    in_types = ['int', 'int']
    out_type = 'transform'
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy
    def accept(self, v): return v.visit_Translate(self.dx, self.dy)

class Recolor(Expr):
    in_types = ['int']
    out_type = 'transform'
    def __init__(self, c): self.c = c
    def accept(self, v): return v.visit_Recolor(self.c)

class Compose(Expr):
    in_types = ['transform', 'transform']
    out_type = 'transform'
    def __init__(self, f, g):
        self.f = f
        self.g = g
    def accept(self, v): return v.visit_Compose(self.f, self.g)

class UnimplementedError(Exception): pass
class Visitor:
    def fail(self, s): raise UnimplementedError(f"`visit_{s}` unimplemented for `{type(self).__name__}`")
    def visit_Nil(self): self.fail('Nil')
    def visit_Num(self, n): self.fail('Num')
    def visit_XMax(self): self.fail('XMax')
    def visit_YMax(self): self.fail('YMax')
    def visit_Z(self, i): self.fail('Z')
    def visit_Not(self, b): self.fail('Not')
    def visit_Plus(self, x, y): self.fail('Plus')
    def visit_Minus(self, x, y): self.fail('Minus')
    def visit_Times(self, x, y): self.fail('Times')
    def visit_Lt(self, x, y): self.fail('Lt')
    def visit_And(self, x, y): self.fail('And')
    def visit_If(self, b, x, y): self.fail('If')
    def visit_Point(self, x, y, color): self.fail('Point')
    def visit_CornerLine(self, x1, y1, x2, y2, color): self.fail('Line')
    def visit_CornerRect(self, x_min, y_min, x_max, y_max, color): self.fail('CornerRect')
    def visit_SizeRect(self, x, y, w, h, color): self.fail('SizeRect')
    def visit_Sprite(self, i, x, y, color): self.fail('Sprite')
    def visit_ColorSprite(self, i, x, y): self.fail('ColorSprite')
    def visit_Join(self, bmp1, bmp2): self.fail('Join')
    def visit_Seq(self, bmps): self.fail('Seq')
    # def visit_Intersect(self, bmp): self.fail('Intersect')
    def visit_HFlip(self): self.fail('HFlip')
    def visit_VFlip(self): self.fail('VFlip')
    def visit_Translate(self, dx, dy): self.fail('Translate')
    def visit_Recolor(self, c): self.fail('Recolor')
    def visit_Compose(self, f, g): self.fail('Compose')
    def visit_Apply(self, f, bmp): self.fail('Apply')
    def visit_Repeat(self, f, n): self.fail('Repeat')

class EnvironmentError(Exception):
    """
    Use this exception to mark errors in Eval caused by random arguments (Z, Sprites)
    """
    pass

class Eval(Visitor):
    def __init__(self, env, height=B_H, width=B_W):
        self.env = env
        self.height = height
        self.width = width

    def make_bitmap(self, f):
        return T.tensor([[f((x, y))
                          for x in range(self.width)]
                          for y in range(self.height)]).float()

    def make_line(self, ax, ay, bx, by, c):
        if ax == bx:  # vertical
            return self.make_bitmap(lambda p: (ax == p[0] and ay <= p[1] <= by) * c)
        elif ay == by:  # horizontal
            return self.make_bitmap(lambda p: (ax <= p[0] <= bx and ay == p[1]) * c)
        elif abs(bx - ax) == abs(by - ay):  # diagonal
            min_x, max_x = (ax, bx) if ax < bx else (bx, ax)
            min_y, max_y = (ay, by) if ay < by else (by, ay)
            return self.make_bitmap(lambda p: (min_x <= p[0] <= max_x and
                                               min_y <= p[1] <= max_y and
                                               p[1] - ay == (ay - by)//(ax - bx) * (p[0] - ax)) * c)
        assert False, "Line must be vertical, horizontal, or diagonal"

    def overlay(self, *bmps):
        def overlay_pt(p):
            x, y = p
            for bmp in bmps:
                if (c := bmp[y][x]) > 0:
                    return c
            return 0
        return self.make_bitmap(overlay_pt)

    def visit_Nil(self):
        return False

    def visit_Num(self, n):
        return n

    def visit_XMax(self):
        return self.width - 1
    
    def visit_YMax(self):
        return self.height - 1

    def visit_Z(self, i):
        assert 'z' in self.env, "Eval env missing Z"
        z = self.env['z'][i]
        return z.item() if isinstance(z, T.Tensor) else z

    def visit_Not(self, b):
        b = b.accept(self)
        assert isinstance(b, bool)
        return not b

    def visit_Plus(self, x, y):
        x, y = x.accept(self), y.accept(self)
        assert isinstance(x, int) and isinstance(y, int)
        return x + y

    def visit_Minus(self, x, y):
        x, y = x.accept(self), y.accept(self)
        assert isinstance(x, int) and isinstance(y, int)
        return x - y

    def visit_Times(self, x, y):
        x, y = x.accept(self), y.accept(self)
        assert isinstance(x, int) and isinstance(y, int)
        return x * y

    def visit_Lt(self, x, y):
        x, y = x.accept(self), y.accept(self)
        assert isinstance(x, int) and isinstance(y, int)
        return x < y

    def visit_And(self, x, y):
        x, y = x.accept(self), y.accept(self)
        assert isinstance(x, bool) and isinstance(y, bool)
        return x and y

    def visit_If(self, b, x, y):
        b, x, y = b.accept(self), x.accept(self), y.accept(self)
        assert isinstance(b, bool) and isinstance(x, int) and isinstance(y, int)
        return x if b else y

    def visit_Point(self, x, y, color):
        x, y, c = x.accept(self), y.accept(self), color.accept(self)
        assert isinstance(x, int) and isinstance(y, int)
        return self.make_bitmap(lambda p: (p[0] == x and p[1] == y) * c)

    def visit_CornerLine(self, x1, y1, x2, y2, color):
        c = color.accept(self)
        x1, y1, x2, y2 = (x1.accept(self), y1.accept(self), x2.accept(self), y2.accept(self))
        assert all(isinstance(v, int) for v in [x1, y1, x2, y2])
        assert abs(x2 - x1) >= 1 or abs(y2 - y1) >= 1
        return self.make_line(x1, y1, x2, y2, c)

    def visit_LengthLine(self, x, y, dx, dy, l, color):
        x, y, dx, dy, l, color = (v.accept(self) for v in [x, y, dx, dy, l, color])
        assert all(isinstance(v, int) for v in [x, y, dx, dy, l])
        assert dx in [-1, 0, 1] and dy in [-1, 0, 1] and not (dx == 0 and dy == 0), \
            f'Found unexpected dx, dy=({dx}, {dy})'
        assert l > 0
        points = sorted([(x, y), (x + dx * (l - 1), y + dy * (l - 1))])
        coords = [v for x, y in points for v in [x, y]]
        return self.make_line(*coords, color)

    def visit_CornerRect(self, x_min, y_min, x_max, y_max, color):
        c = color.accept(self)
        x_min, y_min, x_max, y_max = (x_min.accept(self), y_min.accept(self),
                                      x_max.accept(self), y_max.accept(self))
        assert all(isinstance(v, int) for v in [x_min, y_min, x_max, y_max])
        assert x_min <= x_max and y_min <= y_max
        return self.make_bitmap(lambda p: (x_min <= p[0] <= x_max and y_min <= p[1] <= y_max) * c)

    def visit_SizeRect(self, x, y, w, h, color):
        x, y, w, h, c = (x.accept(self), y.accept(self), w.accept(self), h.accept(self), color.accept(self))
        assert all(isinstance(v, int) for v in [x, y, w, h])
        assert w > 0 and h > 0
        return self.make_bitmap(lambda p: (x <= p[0] < x + w and y <= p[1] < y + h) * c)

    def visit_Sprite(self, i, x, y, color):
        x, y, c = x.accept(self), y.accept(self), color.accept(self)
        return self.translate(self.env['sprites'][i] * c, x, y)

    def visit_ColorSprite(self, i, x, y):
        x, y = x.accept(self), y.accept(self)
        return self.translate(self.env['color-sprites'][i], x, y)

    def visit_Seq(self, bmps):
        bmps = [bmp.accept(self) for bmp in bmps]
        assert all(isinstance(bmp, T.FloatTensor) for bmp in bmps), f"Seq contains unexpected type: {[type(bmp) for bmp in bmps]}"
        return self.overlay(*bmps)

    def visit_Join(self, bmp1, bmp2):
        bmp1, bmp2 = bmp1.accept(self), bmp2.accept(self)
        assert isinstance(bmp1, T.FloatTensor) and isinstance(bmp2, T.FloatTensor), \
            f"Union needs two float tensors, found bmp1={bmp1}, bmp2={bmp2}"
        return self.overlay(bmp1, bmp2)

    # def visit_Intersect(self, bmp1):
    #     bmp1 = bmp1.accept(self)
    #     assert isinstance(bmp1, T.FloatTensor), f"Intersect needs a float tensor, found bmp={bmp1}"
    #     return lambda bmp2: self.make_bitmap(lambda p: int(bmp1[p[1]][p[0]] > 0 and bmp2[p[1]][p[0]] > 0))

    def visit_HFlip(self):
        return lambda bmp: bmp.flip(1)

    def visit_VFlip(self):
        return lambda bmp: bmp.flip(0)

    @staticmethod
    def translate(bmp, dx, dy):
        assert isinstance(bmp, T.Tensor)
        assert isinstance(dx, int) and isinstance(dy, int)

        def slices(delta):
            if delta == 0:  return None, None
            elif delta > 0: return None, -delta
            else:           return -delta, None

        a, b = (dx, 0) if dx > 0 else (0, -dx)
        c, d = (dy, 0) if dy > 0 else (0, -dy)
        c_lo, c_hi = slices(dx)
        r_lo, r_hi = slices(dy)

        return F.pad(bmp[r_lo:r_hi, c_lo:c_hi], (a, b, c, d))

    def visit_Translate(self, dx, dy):
        dx, dy = dx.accept(self), dy.accept(self)
        return lambda bmp: self.translate(bmp, dx, dy)

    def visit_Recolor(self, c):
        def index(bmp, p):
            x, y = p
            return bmp[y][x]

        c = c.accept(self)
        return lambda bmp: self.make_bitmap(lambda p: c if index(bmp, p) > 0 else 0)

    def visit_Compose(self, f, g):
        f, g = f.accept(self), g.accept(self)
        return lambda bmp: f(g(bmp))

    def visit_Repeat(self, f, n):
        n, f = n.accept(self), f.accept(self)
        bmps = []

        def g(bmp):
            for i in range(n):
                bmp = f(bmp)
                bmps.append(bmp)
            return self.overlay(*bmps)

        return g

    def visit_Apply(self, f, bmp):
        # FIXME: the semantics here don't play well with intersection
        f, bmp = f.accept(self), bmp.accept(self)
        return self.overlay(f(bmp), bmp)

class Print(Visitor):
    def __init__(self): pass
    def visit_Nil(self): return 'False'
    def visit_Num(self, n): return f'{n}'
    def visit_XMax(self): return 'x_max'
    def visit_YMax(self): return 'y_max'
    def visit_Z(self, i): return f'z_{i}'
    def visit_Not(self, b): return f'(not {b.accept(self)})'
    def visit_Plus(self, x, y): return f'(+ {x.accept(self)} {y.accept(self)})'
    def visit_Minus(self, x, y): return f'(- {x.accept(self)} {y.accept(self)})'
    def visit_Times(self, x, y): return f'(* {x.accept(self)} {y.accept(self)})'
    def visit_Lt(self, x, y): return f'(< {x.accept(self)} {y.accept(self)})'
    def visit_And(self, x, y): return f'(and {x.accept(self)} {y.accept(self)})'
    def visit_If(self, b, x, y): return f'(if {b.accept(self)} {x.accept(self)} {y.accept(self)})'
    def visit_Point(self, x, y, color):
        return f'(Point[{color.accept(self)}] {x.accept(self)} {y.accept(self)})'
    def visit_CornerLine(self, x1, y1, x2, y2, color):
        return f'(CLine[{color.accept(self)}] {x1.accept(self)} {y1.accept(self)} {x2.accept(self)} {y2.accept(self)})'
    def visit_LengthLine(self, x, y, dx, dy, l, color):
        return f'(LLine[{color.accept(self)}] {x.accept(self)} {y.accept(self)} {dx.accept(self)} {dy.accept(self)} {l.accept(self)})'
    def visit_CornerRect(self, x_min, y_min, x_max, y_max, color):
        return f'(CRect[{color.accept(self)}] {x_min.accept(self)} {y_min.accept(self)} ' \
               f'{x_max.accept(self)} {y_max.accept(self)})'
    def visit_SizeRect(self, x, y, w, h, color):
        return f'(SRect[{color.accept(self)}] {x.accept(self)} {y.accept(self)} ' \
               f'{w.accept(self)} {h.accept(self)})'
    def visit_Sprite(self, i, x, y, color):
        return f'(Sprite_{i}[{color.accept(self)}] {x.accept(self)} {y.accept(self)})'
    def visit_ColorSprite(self, i, x, y):
        return f'(CSprite_{i} {x.accept(self)} {y.accept(self)})'
    def visit_Seq(self, bmps): return '(seq ' + ' '.join([bmp.accept(self) for bmp in bmps]) + ')'
    def visit_Join(self, bmp1, bmp2): return f'(join {bmp1.accept(self)} {bmp2.accept(self)})'
    # def visit_Intersect(self, bmp): return f'(intersect {bmp.accept(self)})'
    def visit_HFlip(self): return 'h-flip'
    def visit_VFlip(self): return 'v-flip'
    def visit_Translate(self, dx, dy): return f'(translate {dx.accept(self)} {dy.accept(self)})'
    def visit_Recolor(self, c): return f'[{c.accept(self)}]'
    def visit_Compose(self, f, g): return f'(compose {f.accept(self)} {g.accept(self)})'
    def visit_Apply(self, f, bmp): return f'({f.accept(self)} {bmp.accept(self)})'
    def visit_Repeat(self, f, n): return f'(repeat {f.accept(self)} {n.accept(self)})'


def deserialize(tokens):
    """
    Deserialize a serialized seq into a program.
    """
    def D(tokens):
        if not tokens: return []
        h, t = tokens[0], D(tokens[1:])
        if isinstance(h, bool) and not h:
            return [Nil()] + t
        if isinstance(h, int):
            return [Num(h)] + t
        if isinstance(h, str):
            if h.startswith('z_'):
                return [Z(int(h[2:]))] + t
            if h.startswith('S_'):
                return [Sprite(int(h[2:]), t[1], t[2], color=t[0])] + t[3:]
            if h.startswith('CS_'):
                return [ColorSprite(int(h[3:]), t[0], t[1])] + t[2:]
            if h == 'x_max':
                return [XMax()] + t
            if h == 'y_max':
                return [YMax()] + t
        if h == '~':
            return [Not(t[0])] + t[1:]
        if h == '+':
            return [Plus(t[0], t[1])] + t[2:]
        if h == '-':
            return [Minus(t[0], t[1])] + t[2:]
        if h == '*':
            return [Times(t[0], t[1])] + t[2:]
        if h == '<':
            return [Lt(t[0], t[1])] + t[2:]
        if h == '&':
            return [And(t[0], t[1])] + t[2:]
        if h == '?':
            return [If(t[0], t[1], t[2])] + t[3:]
        if h == 'P':
            return [Point(t[1], t[2], color=t[0])] + t[3:]
        if h == 'CL':
            return [CornerLine(t[1], t[2], t[3], t[4], color=t[0])] + t[5:]
        if h == 'LL':
            return [LengthLine(t[1], t[2], t[3], t[4], t[5], color=t[0])] + t[6:]
        if h == 'CR':
            return [CornerRect(t[1], t[2], t[3], t[4], color=t[0])] + t[5:]
        if h == 'SR':
            return [SizeRect(t[1], t[2], t[3], t[4], color=t[0])] + t[5:]
        if h == 'H':
            return [HFlip()] + t
        if h == 'V':
            return [VFlip()] + t
        if h == 'T':
            return [Translate(t[0], t[1])] + t[2:]
        if h == '#':
            return [Recolor(t[0])] + t[1:]
        if h == 'o':
            return [Compose(t[0], t[1])] + t[2:]
        if h == '@':
            return [Apply(t[0], t[1])] + t[2:]
        if h == '!':
            return [Repeat(t[0], t[1])] + t[2:]
        # if h == '^':
        #     return [Intersect(t[0])] + t[1:]
        if h == '{':
            i = t.index('}')
            # assert "STOP" in t, f"A sequence must have a STOP token, but none were found: {t}"
            return [Seq(*t[:i])] + t[i + 1:]
        if h == '}':
            return tokens
        else:
            assert False, f'Failed to classify token: {h} of type {type(h)}'

    decoded = D(tokens)
    assert len(decoded) == 1, f'Parsed {len(decoded)} programs in one token sequence, expected one'
    expr = decoded[0]
    assert isinstance(expr, Expr), f'Decoded program should be of type Expr: {expr}'
    assert expr.well_formed(), f'Decoded program should be well-formed: {expr}'
    return expr

class Serialize(Visitor):
    def __init__(self, label_zs=True):
        self.label_zs = label_zs
    def visit_Nil(self): return [False]
    def visit_Num(self, n): return [n]
    def visit_XMax(self): return ['x_max']
    def visit_YMax(self): return ['y_max']
    def visit_Z(self, i): return [f'z_{i}'] if self.label_zs else ['z']
    def visit_Not(self, b): return ['~'] + b.accept(self)
    def visit_Plus(self, x, y): return ['+'] + x.accept(self) + y.accept(self)
    def visit_Minus(self, x, y): return ['-'] + x.accept(self) + y.accept(self)
    def visit_Times(self, x, y): return ['*'] + x.accept(self) + y.accept(self)
    def visit_Lt(self, x, y): return ['<'] + x.accept(self) + y.accept(self)
    def visit_And(self, x, y): return ['&'] + x.accept(self) + y.accept(self)
    def visit_If(self, b, x, y): return ['?'] + b.accept(self) + x.accept(self) + y.accept(self)
    def visit_Point(self, x, y, color): return ['P'] + color.accept(self) + x.accept(self) + y.accept(self)
    def visit_CornerLine(self, x1, y1, x2, y2, color):
        return ['CL'] + color.accept(self) + x1.accept(self) + y1.accept(self) + x2.accept(self) + y2.accept(self)
    def visit_LengthLine(self, x, y, dx, dy, l, color):
        return ['LL'] + color.accept(self) + x.accept(self) + y.accept(self) + dx.accept(self) + dy.accept(self) + l.accept(self)
    def visit_CornerRect(self, x_min, y_min, x_max, y_max, color):
        return ['CR'] + color.accept(self) + x_min.accept(self) + y_min.accept(self) +\
               x_max.accept(self) + y_max.accept(self)
    def visit_SizeRect(self, x, y, w, h, color):
        return ['SR'] + color.accept(self) + x.accept(self) + y.accept(self) + w.accept(self) + h.accept(self)
    def visit_Sprite(self, i, x, y, color):
        return [f'S_{i}'] + color.accept(self) + x.accept(self) + y.accept(self)
    def visit_ColorSprite(self, i, x, y):
        return [f'CS_{i}'] + x.accept(self) + y.accept(self)
    def visit_Seq(self, bmps):
        tokens = ['{']  # start
        for bmp in bmps:
            tokens.extend(bmp.accept(self))
        # tokens.append(SEQ_END)
        tokens.append('}')  # stop
        return tokens
    def visit_Join(self, bmp1, bmp2): return [';'] + bmp1.accept(self) + bmp2.accept(self)
    # def visit_Intersect(self, bmp): return ['^'] + bmp.accept(self)
    def visit_HFlip(self): return ['H']
    def visit_VFlip(self): return ['V']
    def visit_Translate(self, x, y): return ['T'] + x.accept(self) + y.accept(self)
    def visit_Recolor(self, c): return ['#'] + c.accept(self)
    def visit_Compose(self, f, g): return ['o'] + f.accept(self) + g.accept(self)
    def visit_Apply(self, f, bmp): return ['@'] + f.accept(self) + bmp.accept(self)
    def visit_Repeat(self, f, n): return ['!'] + f.accept(self) + n.accept(self)

class SimplifyIndices(Visitor):
    def __init__(self, zs, sprites, csprites):
        """
        zs: the indices of zs in the whole expression
        sprites: the indices of sprites in the whole expression
        """
        self.z_mapping = {z: i for i, z in enumerate(zs)}
        self.sprite_mapping = {sprite: i for i, sprite in enumerate(sprites)}
        self.csprite_mapping = {csprite: i for i, csprite in enumerate(csprites)}
    # Base cases
    def visit_Z(self, i):
        return Z(self.z_mapping[i])
    def visit_Sprite(self, i, x, y, color):
        return Sprite(self.sprite_mapping[i], x.accept(self), y.accept(self), color=color.accept(self))
    def visit_ColorSprite(self, i, x, y):
        return ColorSprite(self.csprite_mapping[i], x.accept(self), y.accept(self))
    
    # Recursive cases
    def visit_Nil(self): return Nil()
    def visit_Num(self, n): return Num(n)
    def visit_XMax(self): return XMax()
    def visit_YMax(self): return YMax()
    def visit_Not(self, b): return Not(b.accept(self))
    def visit_Plus(self, x, y): return Plus(x.accept(self), y.accept(self))
    def visit_Minus(self, x, y): return Minus(x.accept(self), y.accept(self))
    def visit_Times(self, x, y): return Times(x.accept(self), y.accept(self))
    def visit_Lt(self, x, y): return Lt(x.accept(self), y.accept(self))
    def visit_And(self, x, y): return And(x.accept(self), y.accept(self))
    def visit_If(self, b, x, y): return If(b.accept(self), x.accept(self), y.accept(self))
    def visit_Point(self, x, y, color): return Point(x.accept(self), y.accept(self), color.accept(self))
    def visit_CornerLine(self, x1, y1, x2, y2, color):
        return CornerLine(x1.accept(self), y1.accept(self), x2.accept(self), y2.accept(self), color.accept(self))
    def visit_LengthLine(self, x, y, dx, dy, l, color):
        return LengthLine(x.accept(self), y.accept(self), dx.accept(self), dy.accept(self), l.accept(self), color.accept(self))
    def visit_CornerRect(self, x_min, y_min, x_max, y_max, color):
        return CornerRect(x_min.accept(self), y_min.accept(self), x_max.accept(self), y_max.accept(self),
                          color.accept(self))
    def visit_SizeRect(self, x, y, w, h, color):
        return SizeRect(x.accept(self), y.accept(self), w.accept(self), h.accept(self), color.accept(self))
    def visit_Seq(self, bmps): return Seq(*[bmp.accept(self) for bmp in bmps])
    def visit_Join(self, bmp1, bmp2): return Join(bmp1.accept(self), bmp2.accept(self))
    # def visit_Intersect(self, bmp): return Intersect(bmp.accept(self))
    def visit_HFlip(self): return HFlip()
    def visit_VFlip(self): return VFlip()
    def visit_Translate(self, x, y): return Translate(x.accept(self), y.accept(self))
    def visit_Recolor(self, c): return Recolor(c.accept(self))
    def visit_Compose(self, f, g): return Compose(f.accept(self), g.accept(self))
    def visit_Apply(self, f, bmp): return Apply(f.accept(self), bmp.accept(self))
    def visit_Repeat(self, f, n): return Repeat(f.accept(self), n.accept(self))

class WellFormed(Visitor):
    # TODO: clean up exception handling (unsafe as is)
    def __init__(self): pass
    def visit_Nil(self): return True
    def visit_Num(self, n): return isinstance(n, int)
    def visit_XMax(self): return True
    def visit_YMax(self): return True
    def visit_Z(self, i): return isinstance(i, int)
    def visit_Not(self, b): 
        return b.out_type == 'bool' and b.accept(self)
    def visit_Plus(self, x, y):
        return x.out_type == 'int' and y.out_type == 'int' and x.accept(self) and y.accept(self)
    def visit_Minus(self, x, y):
        return x.out_type == 'int' and y.out_type == 'int' and x.accept(self) and y.accept(self)
    def visit_Times(self, x, y):
        return x.out_type == 'int' and y.out_type == 'int' and x.accept(self) and y.accept(self)
    def visit_Lt(self, x, y):
        return x.out_type == 'int' and y.out_type == 'int' and x.accept(self) and y.accept(self)
    def visit_And(self, x, y):
        return x.out_type == 'bool' and y.out_type == 'bool' and x.accept(self) and y.accept(self)
    def visit_If(self, b, x, y):
        # x, y don't have fixed types, but they should have the same type
        return b.out_type == 'bool' and b.accept(self) and \
               x.out_type == y.out_type and x.accept(self) and y.accept(self)
    def visit_Point(self, x, y, color):
        return x.out_type == 'int' and y.out_type == 'int' and color.out_type == 'int' and \
           x.accept(self) and y.accept(self) and color.accept(self)
    def visit_CornerLine(self, x1, y1, x2, y2, color):
        return all(v.out_type == 'int' and v.accept(self) for v in [x1, y1, x2, y2, color])
    def visit_LengthLine(self, x, y, dx, dy, l, color):
        return all(v.out_type == 'int' and v.accept(self) for v in [x, y, dx, dy, l, color])
    def visit_CornerRect(self, x_min, y_min, x_max, y_max, color):
        return all(v.out_type == 'int' and v.accept(self) for v in [x_min, y_min, x_max, y_max, color])
    def visit_SizeRect(self, x, y, w, h, color):
        return all(v.out_type == 'int' and v.accept(self) for v in [x, y, w, h, color])
    def visit_Sprite(self, i, x, y, color):
        return isinstance(i, int) and all(v.out_type == 'int' and v.accept(self) for v in [x, y, color])
    def visit_ColorSprite(self, i, x, y):
        return isinstance(i, int) and all(v.out_type == 'int' and v.accept(self) for v in [x, y])
    def visit_Seq(self, bmps): return all(bmp.out_type == 'bitmap' and bmp.accept(self) for bmp in bmps)
    def visit_Join(self, bmp1, bmp2): return all(bmp.out_type == 'bitmap' and bmp.accept(self) for bmp in [bmp1, bmp2])
    # def visit_Intersect(self, bmp):
    #     return bmp.out_type == 'bitmap' and bmp.accept(self)
    def visit_HFlip(self): return True
    def visit_VFlip(self): return True
    def visit_Translate(self, x, y): 
        return x.out_type == 'int' and y.out_type == 'int' and x.accept(self) and y.accept(self)
    def visit_Recolor(self, c):
        return c.out_type == 'int' and c.accept(self)
    def visit_Compose(self, f, g):
        return f.out_type == 'transform' and g.out_type == 'transform' and f.accept(self) and g.accept(self)
    def visit_Apply(self, f, bmp):
        return f.out_type == 'transform' and bmp.out_type == 'bitmap' and f.accept(self) and bmp.accept(self)
    def visit_Repeat(self, f, n):
        return f.out_type == 'transform' and n.out_type == 'int' and f.accept(self) and n.accept(self)

class Perturb(Visitor):
    def __init__(self, range): self.range = range
    def visit_Nil(self): return Not(Nil())
    def visit_Num(self, n): return Num(n + random.choice([-1, 1]) * random.randint(*self.range))
    def visit_Z(self, i): return Z(random.randint(0, LIB_SIZE - 1))
    def visit_HFlip(self): return VFlip()
    def visit_VFlip(self): return HFlip()
    def visit_Plus(self, x, y):
        return Minus(x, y) if random.randint(0, 1) > 0 else Times(x, y)
    def visit_Minus(self, x, y):
        return Times(x, y) if random.randint(0, 1) > 0 else Plus(x, y) 
    def visit_Times(self, x, y):
        return Plus(x, y) if random.randint(0, 1) > 0 else Minus(x, y)

class MapReduce(Visitor):
    def __init__(self, f_reduce, f_map):
        self.reduce = f_reduce
        self.f = f_map
    
    # Map (apply f)
    def visit_Nil(self): return self.f(Nil)
    def visit_Num(self, n): return self.f(Num, n)
    def visit_XMax(self): return self.f(XMax)
    def visit_YMax(self): return self.f(YMax)
    def visit_Z(self, i): return self.f(Z, i)
    def visit_HFlip(self): return self.f(HFlip)
    def visit_VFlip(self): return self.f(VFlip)

    # Map and reduce
    def visit_Sprite(self, i, x, y, color):
        return self.reduce(Sprite, self.f(Sprite, i), x.accept(self), y.accept(self), color.accept(self))
    def visit_ColorSprite(self, i, x, y):
        return self.reduce(ColorSprite, self.f(ColorSprite, i), x.accept(self), y.accept(self))

    # Reduce
    def visit_Not(self, b): return self.reduce(Not, b.accept(self))
    def visit_Plus(self, x, y): return self.reduce(Plus, x.accept(self), y.accept(self))
    def visit_Minus(self, x, y): return self.reduce(Minus, x.accept(self), y.accept(self))
    def visit_Times(self, x, y): return self.reduce(Times, x.accept(self), y.accept(self))
    def visit_Lt(self, x, y): return self.reduce(Lt, x.accept(self), y.accept(self))
    def visit_And(self, x, y): return self.reduce(And, x.accept(self), y.accept(self))
    def visit_If(self, b, x, y): return self.reduce(If, x.accept(self), y.accept(self))
    def visit_Point(self, x, y, color): return self.reduce(Point, x.accept(self), y.accept(self), color.accept(self))
    def visit_CornerLine(self, x1, y1, x2, y2, color):
        return self.reduce(CornerLine, x1.accept(self), y1.accept(self), x2.accept(self), y2.accept(self), color.accept(self))
    def visit_LengthLine(self, x, y, dx, dy, l, color):
        return self.reduce(LengthLine, x.accept(self), y.accept(self), dx.accept(self), dy.accept(self), l.accept(self), color.accept(self))
    def visit_CornerRect(self, x_min, y_min, x_max, y_max, color):
        return self.reduce(CornerRect, x_min.accept(self), y_min.accept(self), x_max.accept(self), y_max.accept(self),
                           color.accept(self))
    def visit_SizeRect(self, x, y, w, h, color):
        return self.reduce(SizeRect, x.accept(self), y.accept(self), w.accept(self), h.accept(self))
    def visit_Join(self, bmp1, bmp2): return self.reduce(Join, bmp1.accept(self), bmp2.accept(self))
    def visit_Seq(self, bmps): return self.reduce(Seq, *[bmp.accept(self) for bmp in bmps])
    # def visit_Intersect(self, bmp): self.fail('Intersect')
    def visit_Translate(self, dx, dy): return self.reduce(Translate, dx.accept(self), dy.accept(self))
    def visit_Recolor(self, c): return self.reduce(Recolor, c.accept(self))
    def visit_Compose(self, f, g): return self.reduce(Compose, f.accept(self), g.accept(self))
    def visit_Apply(self, f, bmp): return self.reduce(Apply, f.accept(self), bmp.accept(self))
    def visit_Repeat(self, f, n): return self.reduce(Repeat, f.accept(self), n.accept(self))

class Range(Visitor):
    def __init__(self, envs, height=B_H, width=B_W):
        self.envs = envs
        self.height = height
        self.width = width
    def visit_Num(self, n):
        return n, n
    def visit_XMax(self):
        return self.width - 1, self.width - 1
    def visit_YMax(self):
        return self.height - 1, self.height - 1
    def visit_Z(self, i):
        return (min(env['z'][i] for env in self.envs),
                max(env['z'][i] for env in self.envs))
    def visit_Plus(self, x, y):
        x_min, x_max = x.accept(self)
        y_min, y_max = y.accept(self)
        return x_min + y_min, x_max + y_max
    def visit_Minus(self, x, y):
        x_min, x_max = x.accept(self)
        y_min, y_max = y.accept(self)
        return x_min - y_max, x_max - y_min
    def visit_Times(self, x, y):
        x_min, x_max = x.accept(self)
        y_min, y_max = y.accept(self)
        products = [x * y for x in [x_min, x_max] for y in [y_min, y_max]]
        return min(products), max(products)
    def visit_CornerRect(self, x_min, y_min, x_max, y_max, color):
        vals = list(it.chain.from_iterable(v.accept(self) for v in [x_min, y_min, x_max, y_max, color]))
        return min(vals), max(vals)
    def visit_SizeRect(self, x, y, w, h, color):
        vals = list(it.chain.from_iterable(v.accept(self) for v in [x, y, w, h, color]))
        return min(vals), max(vals)

def test_eval():
    tests = [
        # Basic semantics 
        (Nil(),
         lambda z: False),
        (Not(Nil()),
         lambda z: True),
        (Times(Z(0), Z(1)),
         lambda z: z[0] * z[1]),
        (If(Lt(Z(0), Z(1)),
            Z(0),
            Z(1)),
         lambda z: min(z[0], z[1])),
        (If(Not(Lt(Z(0),
                   Z(1))),
            Times(Z(0), Z(1)),
            Plus(Z(0), Z(1))),
         lambda z: z[0] * z[1] if not (z[0] < z[1]) else z[0] + z[1]),
        (CornerRect(Num(0), Num(0),
                    Num(1), Num(2)),
         lambda z: util.img_to_tensor(["##__",
                                       "##__",
                                       "##__",
                                       "____"], w=B_W, h=B_H)),
        (LengthLine(Num(2), Num(3), Num(-1), Num(1), Num(3)),
         lambda z: util.img_to_tensor(["_____",
                                       "_____",
                                       "_____",
                                       "__#__",
                                       "_#___",
                                       "#____",], w=B_W, h=B_H)),
        (SizeRect(Num(0), Num(1), Num(2), Num(2)),
         lambda z: util.img_to_tensor(["____",
                                       "##__",
                                       "##__"], w=B_W, h=B_H)),
        (SizeRect(Num(2), Num(2), Num(1), Num(1)),
         lambda z: util.img_to_tensor(["____",
                                       "____",
                                       "__#_",
                                       "____"], w=B_W, h=B_H)),
        (XMax(),
         lambda z: B_W - 1),
        (YMax(),
         lambda z: B_H - 1),
    ]
    for expr, correct_semantics in tests:
        for x in range(10):
            for y in range(10):
                out = expr.eval({"z": [x, y]})
                expected = correct_semantics([x, y])
                t = expr.out_type
                if t in ['int', 'bool']:
                    assert out == expected, f"failed eval test:\n" \
                                            f" expr=\n{expr}\n" \
                                            f" expected=\n{expected}\n" \
                                            f" out=\n{out}"
                elif t == 'bitmap':
                    assert T.equal(out, expected), f"failed eval test:\n" \
                                                   f" expr=\n{expr}\n" \
                                                   f" expected=\n{expected}\n" \
                                                   f" out=\n{out}"
                else:
                    assert False, "type error in eval test"

    # (0,0), (1,1)
    expr = CornerRect(Num(0), Num(0),
                      Num(1), Num(1))
    out = expr.eval({'z': []})
    expected = util.img_to_tensor(["##__",
                                   "##__",
                                   "____",
                                   "____"], w=B_W, h=B_H)
    assert T.equal(expected, out), f"test_render failed:\n expected={expected},\n out={out}"

    # (1,0), (3,3)
    expr = CornerRect(Z(0), Num(0),
                      Plus(Z(0), Num(2)), Num(3))
    out = expr.eval({'z': [1, 2, 3]})
    expected = util.img_to_tensor(["_###",
                                   "_###",
                                   "_###",
                                   "_###"], w=B_W, h=B_H)
    assert T.equal(expected, out), f"test_render failed:\n expected={expected},\n out={out}"
    print(" [+] passed test_eval")

def test_eval_bitmap():
    tests = [
        # Line tests
        (CornerLine(Num(0), Num(0),
                    Num(1), Num(1)),
         ["#___",
          "_#__",
          "____",
          "____"]),
        (CornerLine(Num(0), Num(0),
                    Num(3), Num(3)),
         ["#___",
          "_#__",
          "__#_",
          "___#"]),
        (CornerLine(Num(1), Num(0),
                    Num(3), Num(2)),
         ["_#__",
          "__#_",
          "___#",
          "____"]),
        (CornerLine(Num(1), Num(2),
                    Num(2), Num(3)),
         ["____",
          "____",
          "_#__",
          "__#_"]),
        (CornerLine(Num(1), Num(0),
                    Num(3), Num(0)),
         ["_###",
          "____",
          "____",
          "____"]),
        (CornerLine(Num(1), Num(2),
                    Num(1), Num(3)),
         ["____",
          "____",
          "_#__",
          "_#__"]),
        (LengthLine(Num(0), Num(0), Num(1), Num(1), Num(3)),
         ["#__",
          "_#_",
          "__#"]),
        (LengthLine(Num(0), Num(0), Num(1), Num(1), Num(2)),
         ["#__",
          "_#_",
          "___"]),
        (LengthLine(Num(1), Num(0), Num(0), Num(1), Num(3)),
         ["_#_",
          "_#_",
          "_#_"]),
        (LengthLine(Num(1), Num(0), Num(0), Num(1), Num(5)),
         ["_#_",
          "_#_",
          "_#_",
          "_#_",
          "_#_",]),
        (LengthLine(Num(3), Num(2), Num(1), Num(-1), Num(2)),
         ["_______",
          "____#__",
          "___#___",
          "_______",]),
        (LengthLine(Num(3), Num(2), Num(-1), Num(1), Num(2)),
         ["_______",
          "_______",
          "___#___",
          "__#____",]),
        (LengthLine(Num(3), Num(2), Num(-1), Num(-1), Num(2)),
         ["_______",
          "__#____",
          "___#___",
          "_______",]),
        
        # Reflection
        (Apply(HFlip(),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["#___" + "_"*(B_W-8) + "___#",
          "_#__" + "_"*(B_W-8) + "__#_",
          "__#_" + "_"*(B_W-8) + "_#__",
          "___#" + "_"*(B_W-8) + "#___"]),
        (Join(Join(Point(Num(0), Num(0)),
                   Point(Num(1), Num(3))),
              Join(Point(Num(2), Num(0)),
                   Point(Num(3), Num(1)))),
         ["#_#_",
          "___#",
          "____",
          "_#__"]),
        (Apply(VFlip(),
               Join(Join(Point(Num(0), Num(0)),
                         Point(Num(1), Num(3))),
                    Join(Point(Num(2), Num(0)),
                         Point(Num(3), Num(1))))),
         ["#_#_",
          "___#",
          "____",
          "_#__"] +
         ["____"] * (B_H - 8) +
         ["_#__",
          "____",
          "___#",
          "#_#_"]),

        # Joining
        (Join(CornerRect(Num(0), Num(0),
                         Num(1), Num(1)),
              CornerLine(Num(2), Num(3),
                         Num(3), Num(3))),
         ["##__",
          "##__",
          "____",
          "__##"]),
        (Apply(HFlip(),
               Join(CornerRect(Num(0), Num(0),
                               Num(1), Num(1)),
                    CornerRect(Num(2), Num(2),
                               Num(3), Num(3)))),
         ["##__" + "_"*(B_W-8) + "__##",
          "##__" + "_"*(B_W-8) + "__##",
          "__##" + "_"*(B_W-8) + "##__",
          "__##" + "_"*(B_W-8) + "##__"]),

        # Translate
        (Apply(Translate(Num(0), Num(0)),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["#___",
          "_#__",
          "__#_",
          "___#"]),
        (Apply(Compose(Translate(Num(1), Num(0)), Recolor(Num(2))),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["12___",
          "_12__",
          "__12_",
          "___12"]),
        (Apply(Compose(Translate(Num(-1), Num(0)), Recolor(Num(2))),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["1____",
          "21___",
          "_21__",
          "__21_"]),
        (Apply(Compose(Translate(Num(0), Num(1)), Recolor(Num(2))),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["1___",
          "21__",
          "_21_",
          "__21",
          "___2"]),
        (Apply(Compose(Translate(Num(0), Num(-1)), Recolor(Num(2))),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["12___",
          "_12__",
          "__12_",
          "___1_"]),
        (Apply(Compose(Translate(Num(-1), Num(-1)), Recolor(Num(2))),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["2___",
          "_2__",
          "__2_",
          "___1"]),
        (Apply(Compose(Translate(Num(1), Num(1)), Recolor(Num(2))),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["1____",
          "_2___",
          "__2__",
          "___2_",
          "____2"]),
        (Apply(Compose(Translate(Num(2), Num(3)), Recolor(Num(2))),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["1_____",
          "_1____",
          "__1___",
          "__21__",
          "___2__",
          "____2_",
          "_____2"]),
        (Apply(Repeat(Translate(Num(1), Num(1)), Num(5)),
               Point(Num(0), Num(0))),
         ["#_____",
          "_#____",
          "__#___",
          "___#__",
          "____#_",
          "_____#"]),
        (Apply(Repeat(Translate(Num(2), Num(0)), Num(2)),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["#_#_#___",
          "_#_#_#__",
          "__#_#_#_",
          "___#_#_#"]),
        (Apply(Repeat(Compose(Translate(Num(2), Num(0)),
                              Recolor(Num(2))),
                      Num(2)),
               CornerLine(Num(0), Num(0),
                          Num(3), Num(3))),
         ["1_2_2___",
          "_1_2_2__",
          "__1_2_2_",
          "___1_2_2"]),
        (CornerRect(Num(0), Num(0), Num(2), Num(2)),
         ["###_",
          "###_",
          "###_",
          "____"]),
        (CornerLine(Num(0), Num(0), Num(3), Num(3)),
         ["#___",
          "_#__",
          "__#_",
          "___#"]),
        # (Apply(Intersect(Rect(Num(0), Num(0), Num(2), Num(2))),
        #        Line(Num(0), Num(0), Num(3), Num(3))),
        #  ["#__",
        #   "_#_",
        #   "__#"]),
    ]
    for expr, correct_semantics in tests:
        out = expr.eval({"z": []})
        expected = util.img_to_tensor(correct_semantics, w=B_W, h=B_H)
        assert T.equal(out, expected), \
            f"failed eval test:\n" \
            f" expr=\n{expr}\n" \
            f" expected=\n{expected}\n" \
            f" out=\n{out}"
    print(" [+] passed test_eval_bitmap")

def test_eval_sprite():
    tests = [
        ([["1___",
           "1___",
           "____",
           "____"]],
         Sprite(0),
         ["1___",
          "1___",
          "____",
          "____"]),
        ([["11",
           "1_"],
          ["11",
           "_1"]],
         Sprite(0, color=Num(4)),
         ["44",
          "4_"]),
        ([["1",
           "1_"],
          ["11",
           "_1"]],
         Sprite(1),
         ["11",
          "_1"]),
        ([["1",
           "1_"],
          ["11",
           "_1"]],
         Sprite(1, x=Num(1), y=Num(2)),
         ["___",
          "___",
          "_11",
          "__1"]),
        ([["111",
           "__1",
           "_1_"]],
         Apply(Compose(HFlip(),
                       Recolor(Num(2))),
               Sprite(0)),
            ["111" + '_' * (B_W - 6) + "222",
             "__1" + '_' * (B_W - 6) + "2__",
             "_1_" + '_' * (B_W - 6) + "_2_"]),
        ([["111",
           "__1",
           "_1_"]],
         Apply(Compose(VFlip(),
                       Recolor(Num(2))),
               Sprite(0)),
         ["111",
          "__1",
          "_1_"]
         +
         ["___"] * (B_H - 6)
         +
         ["_2_",
          "__2",
          "222"]),
    ]
    for sprites, expr, correct_semantics in tests:
        env = {'z': [],
               'sprites': [util.img_to_tensor(s, w=B_W, h=B_H) for s in sprites]}
        out = expr.eval(env)
        expected = util.img_to_tensor(correct_semantics, w=B_W, h=B_H)
        assert T.equal(out, expected), \
            f"failed test:\n" \
            f" expr=\n{expr}\n" \
            f" expected=\n{expected}\n" \
            f" out=\n{out}"
    print(" [+] passed test_sprite")

def test_eval_colorsprite():
    tests = [
        ([["12_2",
           "1_35",
           "_45_",]],
         ColorSprite(0),
         ["12_2",
           "1_35",
           "_45_",]),
        ([["21",
           "1_"],
          ["12",
           "_1"]],
         ColorSprite(1),
         ["12",
          "_1"]),
        ([["1",
           "2"],
          ["12",
           "21"]],
         ColorSprite(1, x=Num(1), y=Num(2)),
         ["___",
          "___",
          "_12",
          "_21"]),
    ]
    for sprites, expr, correct_semantics in tests:
        env = {'z': [],
               'sprites': [],
               'color-sprites': [util.img_to_tensor(s, w=B_W, h=B_H) for s in sprites]}
        out = expr.eval(env)
        expected = util.img_to_tensor(correct_semantics, w=B_W, h=B_H)
        assert T.equal(out, expected), \
            f"failed test:\n" \
            f" expr=\n{expr}\n" \
            f" expected=\n{expected}\n" \
            f" out=\n{out}"
    print(" [+] passed test_csprite")

def test_eval_color():
    tests = [
        (CornerRect(Num(0), Num(0),
                    Num(1), Num(1), Num(2)),
         ["22__",
          "22__",
          "____",
          "____"]),
        (CornerLine(Num(1), Num(0),
                    Num(3), Num(2), Num(3)),
         ["_3__",
          "__3_",
          "___3",
          "____"]),
        (CornerLine(Num(1), Num(0),
                    Num(3), Num(0), Num(2)),
         ["_222",
          "____",
          "____",
          "____"]),
        (Join(Join(Point(Num(0), Num(0), Num(2)),
                   Point(Num(1), Num(3), Num(3))),
              Join(Point(Num(2), Num(0), Num(7)),
                   Point(Num(3), Num(1), Num(9)))),
         ["2_7_",
          "___9",
          "____",
          "_3__"]),
        (Join(CornerRect(Num(0), Num(0),
                         Num(1), Num(1), Num(1)),
              CornerRect(Num(2), Num(2),
                         Num(3), Num(3), Num(6))),
         ["11__",
          "11__",
          "__66",
          "__66"]),
    ]
    for expr, correct_semantics in tests:
        out = expr.eval({"z": []})
        expected = util.img_to_tensor(correct_semantics, w=B_W, h=B_H)
        assert T.equal(out, expected), \
            f"failed eval color test:\n" \
            f" expr=\n{expr}\n" \
            f" expected=\n{expected}\n" \
            f" out=\n{out}"
    print(" [+] passed test_eval_color")

def test_zs():
    test_cases = [
        (CornerRect(Num(0), Num(1), Num(2), Num(3)),
         []),
        (CornerRect(Z(0), Z(1), Plus(Z(0), Num(3)), Plus(Z(1), Num(3))),
         [0, 1]),
        (CornerRect(Z(3), Z(1), Z(3), Z(0)),
         [3, 1, 0]),
        (CornerRect(Z(0), Z(1), Z(3), Z(1)),
         [0, 1, 3]),
    ]
    for expr, ans in test_cases:
        out = expr.zs()
        assert out == ans, f"test_zs failed: expected={ans}, actual={out}"
    print(" [+] passed test_zs")

def test_sprites():
    test_cases = [
        (Num(0), []),
        (Sprite(0), [0]),
        (Seq(Sprite(0), Sprite(1), CornerRect(Num(0), Num(0), Num(2), Num(2))),
         [0, 1]),
        (Seq(Sprite(1), Sprite(0), Sprite(2), CornerRect(Num(0), Num(0), Num(2), Num(2))),
         [1, 0, 2]),
    ]
    for expr, ans in test_cases:
        out = expr.sprites()
        assert out == ans, f"test_sprites failed: expected={ans}, actual={out}"
    print(" [+] passed test_sprites")

def test_simplify_indices():
    test_cases = [
        (Seq(Z(0), Z(1), Z(3)),
         Seq(Z(0), Z(1), Z(2))),
        (Seq(Z(7), Z(9), Z(3)),
         Seq(Z(0), Z(1), Z(2))),
        (CornerRect(Z(2), Z(1), Plus(Z(0), Z(2)), Plus(Z(1), Z(3))),
         CornerRect(Z(0), Z(1), Plus(Z(2), Z(0)), Plus(Z(1), Z(3)))),
        (Seq(Sprite(1), Sprite(0), Sprite(2), Sprite(0), Sprite(1), Z(3), Z(3)),
         Seq(Sprite(0), Sprite(1), Sprite(2), Sprite(1), Sprite(0), Z(0), Z(0))),
    ]
    for expr, ans in test_cases:
        out = expr.simplify_indices()
        assert out == ans, f"test_simplify_indices failed: expected={ans}, actual={out}"
    print(" [+] passed test_simplify_indices")

def test_serialize():
    test_cases = [
        (Nil(), [False]),
        (XMax(), ['x_max']),
        (Plus(Z(0), Z(1)), ['+', 'z_0', 'z_1']),
        (Sprite(0, color=Num(7)), ['S_0', 7, 0, 0]),
        (Plus(Times(Num(1), Num(0)), Minus(Num(3), Num(2))), ['+', '*', 1, 0, '-', 3, 2]),
        (And(Not(Nil()), Nil()), ['&', '~', False, False]),
        (Not(Lt(Num(3), Minus(Num(2), Num(7)))), ['~', '<', 3, '-', 2, 7]),
        (If(Not(Lt(Num(3), Z(0))), Num(2), Num(5)), ['?', '~', '<', 3, 'z_0', 2, 5]),
        (If(Lt(Z(0), Z(1)),
            Point(Z(0), Z(0)),
            CornerRect(Z(1), Z(1), Num(2), Num(3))),
         ['?', '<', 'z_0', 'z_1',
          'P', 1, 'z_0', 'z_0',
          'CR', 1, 'z_1', 'z_1', 2, 3]),
        (Seq(Sprite(0), Sprite(1, color=Num(6)), Sprite(2), Sprite(3)),
         ['{',
          'S_0', 1, 0, 0,
          'S_1', 6, 0, 0,
          'S_2', 1, 0, 0,
          'S_3', 1, 0, 0,
          '}']),
        (Apply(Translate(Num(1), Num(2)),
               Seq(CornerRect(Plus(Z(0), Num(1)),
                              Plus(Z(0), Num(1)),
                              Num(2),
                              Num(2)),
                   CornerRect(Z(0), Z(0), Num(2), Num(2)))),
         ['@', 'T', 1, 2, '{',
          'CR', 1, '+', 'z_0', 1, '+', 'z_0', 1, 2, 2,
          'CR', 1, 'z_0', 'z_0', 2, 2, '}']),
    ]
    for expr, ans in test_cases:
        serialized = expr.serialize()
        deserialized = deserialize(serialized)
        assert serialized == ans, \
            f'serialization failed: in={expr}:\n  expected {ans},\n   but got {serialized}'
        assert deserialized == expr, \
            f'deserialization failed: in={expr}:\n  expected {expr},\n   but got {deserialized}'
    print(' [+] passed test_serialize')

def test_deserialize_breaking():
    test_cases = [
        ([1], False),
        ([1, 2, 3], True),
        (['{'], True),
        (['P', 0, 1, 2], False),
        (['P', 'g', 1, 2], True),
        (['P', 'CR', 0, 1, 2, 3, 4, 5, 6], True),
        (['{', 'P', 0, 1, 2, '}'], False),
        (['CL', 1, 1, 3, 3, 2], False),
        (['CL', 'g', 1, 1, 3, 3], True),
        (['CR', 0, 9, 'CR', 11, 6, 8, '}', '}', 4, 2, 8, 15, 9, 9, 7, 13, 4, '}', 2, 8], True),
        (['CL', 'CR', 4, 8, 2, 4, 3, 2, '}', 9, 1, '}', 2, 6, '}', 6, 4, 8], True),
    ]
    for case, should_fail in test_cases:
        try:
            out = deserialize(case)
            failed = False
        except (AssertionError, ValueError):
            failed = True

        if should_fail and not failed:
            print( f"expected to fail but didn't: in={case}, got {out}")
            exit(1)
        elif not should_fail and failed:
            print( f"failed unexpectedly: in={case}")
            exit(1)
    print(" [+] passed test_deserialize_breaking")
    
def test_well_formed():
    test_cases = [
        (XMax(), True),
        (Point(Num(0), Num(1)), True),
        (Point(0, 1), False),
        (CornerLine(Num(1), Num(1), Num(3), Num(3), Num(1)), True),
    ]
    for expr, ans in test_cases:
        out = expr.well_formed()
        assert out == ans, f'well_formed case failed: in={expr}, expected={ans}, got={out}'
    print(' [+] passed test_well_formed')

def test_range():
    envs = [
        {'z': [1, 0]},
        {'z': [-3, 3]},
        {'z': [2, 5]},
        {'z': [8, -4]},
    ]
    test_cases = [
        (Num(0), 0, 0),
        (Num(1), 1, 1),
        (Z(0), -3, 8),
        (Z(1), -4, 5),
        (Plus(Z(0), Z(1)), -7, 13),
        (Minus(Z(0), Z(1)), -8, 12),
        (Times(Z(0), Z(1)), -32, 40),
        (Times(XMax(), Z(1)), (B_W - 1) * -4, (B_W - 1) * 5),
    ]
    for expr, lo, hi in test_cases:
        out = expr.range(envs)
        assert out == (lo, hi), f"test_range failed: in={expr}, expected={(lo, hi)}, actual={out}"
    print(" [+] passed test_range")

def test_leaves():
    cases = [
        (Num(0), [[Num(0)]]),
        (Plus(Num(0), Num(1)), [[Plus, Num(0)], [Plus, Num(1)]]),
        (Times(Num(1), Num(1)), [[Times, Num(1)], [Times, Num(1)]]),
        (Plus(Times(Num(3), Num(2)),
              Minus(Num(3), Num(1))),
         [[Plus, Times, Num(3)],
          [Plus, Times, Num(2)],
          [Plus, Minus, Num(3)],
          [Plus, Minus, Num(1)]]),
    ]
    for expr, ans in cases:
        leaves = expr.leaves()
        n_leaves = expr.count_leaves()
        assert n_leaves == len(ans), f"count_leaves failed: in={expr}, expected={len(ans)}, actual={n_leaves}"
        assert leaves == ans, f"leaves failed: in={expr}, expected={ans}, actual={leaves}"
    print(" [+] passed test_leaves")

def test_eval_variable_sizes():
    cases = [
        (CornerRect(Num(0), Num(0), Num(1), Num(1)), 6, 6,
         util.img_to_tensor(["##____",
                             "##____",
                             "______",
                             "______",
                             "______",
                             "______",], h=6, w=6)),
        (CornerRect(Num(0), Num(0), Num(2), Num(2)), 3, 3,
         util.img_to_tensor(["###",
                             "###",
                             "###",], h=3, w=3)),
        # should fail with assertion error:
        # (Rect(Num(0), Num(0), Num(2), Num(2)), 1, 1,
        #  util.img_to_tensor(["###",
        #                      "###",
        #                      "###", ], h=3, w=3))
    ]
    for expr, h, w, ans in cases:
        render = expr.eval(height=h, width=w)
        assert T.equal(render, ans), f"Expected={ans}, but got {render}"
    print(" [+] passed test_eval_variable_sizes")

def test_eval_using_xy_max():
    cases = [
        (CornerRect(Num(0), Num(0), XMax(), YMax()), 6, 6,
         util.img_to_tensor(["######",
                             "######",
                             "######",
                             "######",
                             "######",
                             "######",], h=6, w=6)),
        (CornerRect(Num(0), Num(0), XMax(), YMax()), 3, 3,
         util.img_to_tensor(["###",
                             "###",
                             "###",], h=3, w=3)),
        (CornerRect(Num(1), Num(1), XMax(), YMax()), 3, 3,
         util.img_to_tensor(["___",
                             "_##",
                             "_##",], h=3, w=3)),
        (CornerRect(Num(1), Num(1), XMax(), YMax()), 5, 5,
         util.img_to_tensor(["_____",
                             "_####",
                             "_####",
                             "_####",
                             "_####",], h=5, w=5)),
    ]
    for expr, h, w, ans in cases:
        render = expr.eval(height=h, width=w)
        assert T.equal(render, ans), f"Expected={ans}, but got {render}"
    print(" [+] passed test_eval_using_xy_max")

def demo_perturb_leaves():
    cases = [
        Num(0),
        Plus(Num(0), Num(1)),
        Times(Num(1), Num(1)),
        Plus(Times(Num(3), Num(2)), Minus(Num(3), Num(1))),
        CornerRect(Num(0), Num(1), Num(3), Num(3)),
    ]
    for expr in cases:
        size = expr.count_leaves()
        out = expr.perturb_leaves(1)
        print(expr, out)
        # assert out != expr, f"perturb_leaves failed: in={expr}, out={out}"

if __name__ == '__main__':
    test_eval()
    test_eval_bitmap()
    test_eval_color()
    test_eval_sprite()
    test_eval_colorsprite()
    test_sprites()
    test_simplify_indices()
    test_range()
    test_zs()
    test_serialize()
    test_well_formed()
    test_deserialize_breaking()
    test_leaves()
    test_eval_variable_sizes()
    test_eval_using_xy_max()
    # demo_perturb_leaves()
