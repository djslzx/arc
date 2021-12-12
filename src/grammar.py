import util
import nltk
import torch as T
import torch.nn.functional as F
import random
import ant
from math import log2

# bitmap size constants
B_W = 10
B_H = 10

LIB_SIZE = 10
Z_LO = 0  # min poss value in z_n
Z_HI = max(B_W, B_H) - 1  # max poss value in z_n

class Grammar:
    def __init__(self, ops, consts):
        self.ops = ops
        self.consts = consts


class Visited:
    def accept(self, visitor):
        assert False, f"`accept` not implemented for {type(self).__name__}"

    def eval(self, env={}): return self.accept(Eval(env))

    def zs(self): return self.accept(Zs())

    def serialize(self): return self.accept(Serialize())

    def deserialize(self): return deserialize(self)

    def __len__(self): return self.accept(Size())

    def __str__(self): return self.accept(Print())


class Expr(Visited):
    def accept(self, visitor): assert False, f"not implemented for {type(self).__name__}"

    def __repr__(self): return str(self)

    def __eq__(self, other): return str(self) == str(other)

    def __hash__(self): return hash(str(self))

    def __ne__(self, other): return str(self) != str(other)

    def __gt__(self, other): return str(self) > str(other)

    def __lt__(self, other): return str(self) < str(other)

    def dist_to(self, other): 
        '''Edit distance to transform self into other'''
        # TODO: make this better
        return nltk.edit_distance(str(self), str(other))

def seed_zs():
    return (T.rand(LIB_SIZE) * (Z_HI - Z_LO) - Z_LO).long()

def seed_sprites():
    sprites = []
    while len(sprites) < LIB_SIZE:
        sprite = ant.ant(x0=0, y0=0, 
                         w=random.randint(2, B_W-1), 
                         h=random.randint(2, B_H-1), 
                         W=B_W, H=B_H)
        if ant.classify(sprite) == 'Sprite':
            sprites.append(sprite)
        
    return T.stack(sprites)


class Empty(Expr):
    in_types = []
    out_type = 'None'

    def __init__(self): pass

    def accept(self, v): return v.visit_Empty()


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


class Line(Expr):
    in_types = ['int', 'int', 'int', 'int', 'int']
    out_type = 'bitmap'

    def __init__(self, x1, y1, x2, y2, color=Num(1)):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color

    def accept(self, v): return v.visit_Line(self.x1, self.y1, self.x2, self.y2, self.color)


class Point(Expr):
    in_types = ['int', 'int', 'int']
    out_type = 'bitmap'

    def __init__(self, x, y, color=Num(1)):
        self.x = x
        self.y = y
        self.color = color

    def accept(self, v): return v.visit_Point(self.x, self.y, self.color)


class Rect(Expr):
    in_types = ['int', 'int', 'int', 'int', 'int']
    out_type = 'bitmap'

    def __init__(self, x1, y1, x2, y2, color=Num(1)):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color

    def accept(self, v): return v.visit_Rect(self.x1, self.y1, self.x2, self.y2, self.color)


class Sprite(Expr):
    in_types = ['int', 'int', 'int', 'int']
    out_type = 'bitmap'

    def __init__(self, i, x=Num(0), y=Num(0)):
        self.i = i
        self.x = x
        self.y = y

    def accept(self, v): return v.visit_Sprite(self.i, self.x, self.y)


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


class Intersect(Expr):
    in_types = ['bitmap']
    out_type = 'transform'

    def __init__(self, bmp):
        self.bmp = bmp

    def accept(self, v): return v.visit_Intersect(self.bmp)


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

        
class Visitor:

    def fail(self, s): assert False, f"Visitor subclass `{type(self).__name__}` should implement `visit_{s}`"

    def visit_Empty(self): self.fail('Empty')

    def visit_Nil(self): self.fail('Nil')

    def visit_Num(self, n): self.fail('Num')

    def visit_Z(self, i): self.fail('Z')

    def visit_Not(self, b): self.fail('Not')

    def visit_Plus(self, x, y): self.fail('Plus')

    def visit_Minus(self, x, y): self.fail('Minus')

    def visit_Times(self, x, y): self.fail('Times')

    def visit_Lt(self, x, y): self.fail('Lt')

    def visit_And(self, x, y): self.fail('And')

    def visit_If(self, b, x, y): self.fail('If')

    def visit_Point(self, x, y, color): self.fail('Point')

    def visit_Line(self, x1, y1, x2, y2, color): self.fail('Line')

    def visit_Rect(self, x1, y1, x2, y2, color): self.fail('Rect')

    def visit_Sprite(self, i, x, y): self.fail('Sprite')

    def visit_Join(self, bmp1, bmp2): self.fail('Join')

    def visit_Seq(self, bmps): self.fail('Seq')

    def visit_Intersect(self, bmp): self.fail('Intersect')

    def visit_HFlip(self): self.fail('HFlip')

    def visit_VFlip(self): self.fail('VFlip')

    def visit_Translate(self, bmp, dx, dy): self.fail('Translate')

    def visit_Recolor(self, c): self.fail('Recolor')

    def visit_Compose(self, f, g): self.fail('Compose')

    def visit_Apply(self, f, bmp): self.fail('Apply')

    def visit_Repeat(self, f, n): self.fail('Repeat')


class Eval(Visitor):
    def __init__(self, env):
        self.env = env

    @staticmethod
    def make_bitmap(f):
        return T.tensor([[f((x, y))
                          for x in range(B_W)]
                         for y in range(B_H)]).float()

    @staticmethod
    def overlay(*bmps):
        def overlay_pt(p):
            x, y = p
            for bmp in bmps:
                if (c := bmp[y][x]) > 0:
                    return c
            return 0
        return Eval.make_bitmap(overlay_pt)

    def visit_Empty(self):
        return None

    def visit_Nil(self):
        return False

    def visit_Num(self, n):
        return n

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
        return Eval.make_bitmap(lambda p: (p[0] == x and p[1] == y) * c)

    def visit_Line(self, x1, y1, x2, y2, color):
        c = color.accept(self)
        x1, y1, x2, y2 = (x1.accept(self), y1.accept(self),
                          x2.accept(self), y2.accept(self))
        assert all(isinstance(v, int) for v in [x1, y1, x2, y2])
        assert 0 <= x1 <= x2 < B_W and 0 <= y1 <= y2 < B_H
        assert abs(x2 - x1) >= 1 or abs(y2 - y1) >= 1
        if x1 == x2:            # vertical
            return Eval.make_bitmap(lambda p: (x1 == p[0] and y1 <= p[1] <= y2) * c)
        elif y1 == y2:          # horizontal
            return Eval.make_bitmap(lambda p: (x1 <= p[0] <= x2 and y1 == p[1]) * c)
        elif abs(x2 - x1) == abs(y2 - y1): # diagonal
            return Eval.make_bitmap(lambda p: (x1 <= p[0] <= x2 and
                                               y1 <= p[1] <= y2 and
                                               p[1] == y1 + (p[0] - x1)) * c)
        assert False, "Line must be vertical, horizontal, or diagonal"

    def visit_Rect(self, x1, y1, x2, y2, color):
        c = color.accept(self)
        x1, y1, x2, y2 = (x1.accept(self), y1.accept(self),
                          x2.accept(self), y2.accept(self))
        assert all(isinstance(v, int) for v in [x1, y1, x2, y2])
        assert 0 <= x1 <= x2 < B_W and 0 <= y1 <= y2 < B_H
        assert x2 - x1 >= 1 and y2 - y1 >= 1
        return Eval.make_bitmap(lambda p: (x1 <= p[0] <= x2 and y1 <= p[1] <= y2) * c)

    def visit_Sprite(self, i, x, y):
        x, y = x.accept(self), y.accept(self)
        return Eval.translate(self.env['sprites'][i], x, y)

    def visit_Seq(self, bmps):
        bmps = [bmp.accept(self) for bmp in bmps]
        assert all(isinstance(bmp, T.FloatTensor) for bmp in bmps), f"Seq contains unexpected type: {[type(bmp) for bmp in bmps]}"
        return Eval.overlay(*bmps)

    def visit_Join(self, bmp1, bmp2):
        bmp1, bmp2 = bmp1.accept(self), bmp2.accept(self)
        assert isinstance(bmp1, T.FloatTensor) and isinstance(bmp2, T.FloatTensor), \
            f"Union needs two float tensors, found bmp1={bmp1}, bmp2={bmp2}"
        return Eval.overlay(bmp1, bmp2)

    def visit_Intersect(self, bmp1):
        bmp1 = bmp1.accept(self)
        assert isinstance(bmp1, T.FloatTensor), \
            f"Intersect needs a float tensor, found bmp={bmp}"
        def intersect(pt, bmp2):
            x, y = pt
            c1, c2 = bmp1[y][x], bmp2[y][x]
            return (c1 and c2) * c1

        return lambda bmp: Eval.make_bitmap(lambda p: intersect(p, bmp))

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
        dx, dy = dx.eval({}), dy.eval({})
        return lambda bmp: Eval.translate(bmp, dx, dy)

    def visit_Recolor(self, c): 
        def index(bmp, p):
            x, y = p
            return bmp[y][x]

        c = c.accept(self)
        return lambda bmp: Eval.make_bitmap(lambda p: c if index(bmp, p) > 0 else 0)

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
            return Eval.overlay(*bmps)
        return g

    def visit_Apply(self, f, bmp): 
        f, bmp = f.accept(self), bmp.accept(self)
        return Eval.overlay(f(bmp), bmp)


class Size(Visitor):
    def __init__(self): pass

    def visit_Empty(self): return 0

    def visit_Nil(self): return 1

    def visit_Num(self, n): return 1

    def visit_Z(self, i): return 1

    def visit_Not(self, b): return b.accept(self) + 1

    def visit_Plus(self, x, y): return x.accept(self) + y.accept(self) + 1

    def visit_Minus(self, x, y): return x.accept(self) + y.accept(self) + 1

    def visit_Times(self, x, y): return x.accept(self) + y.accept(self) + 1

    def visit_Lt(self, x, y): return x.accept(self) + y.accept(self) + 1

    def visit_And(self, x, y): return x.accept(self) + y.accept(self) + 1

    def visit_If(self, b, x, y): return b.accept(self) + x.accept(self) + y.accept(self) + 1

    def visit_Point(self, x, y, color): return x.accept(self) + y.accept(self) + color.accept(self) + 1

    def visit_Line(self, x1, y1, x2, y2, color): 
        return x1.accept(self) + y1.accept(self) + x2.accept(self) + y2.accept(self) + color.accept(self) + 1

    def visit_Rect(self, x1, y1, x2, y2, color):
        return x1.accept(self) + y1.accept(self) + x2.accept(self) + y2.accept(self) + color.accept(self) + 1

    def visit_Sprite(self, i, x, y):
        return x.accept(self) + y.accept(self) + 1

    def visit_Stack(self, bmps): return sum(bmp.accept(self) for bmp in bmps) + 1

    def visit_Join(self, bmp1, bmp2): return bmp1.accept(self) + bmp2.accept(self) + 1

    def visit_Intersect(self, bmp): return bmp.accept(self) + 1

    def visit_HFlip(self): return 1

    def visit_VFlip(self): return 1

    def visit_Translate(self, x, y): return x.accept(self) + y.accept(self) + 1

    def visit_Recolor(self, c): return c.accept(self) + 1

    def visit_Compose(self, f, g): return f.accept(self) + g.accept(self) + 1

    def visit_Apply(self, f, bmp): return f.accept(self) + bmp.accept(self) + 1

    def visit_Repeat(self, f, n): return f.accept(self) + n.accept(self) + 1


class Print(Visitor):
    def __init__(self): pass

    def visit_Empty(self): return 'Empty'

    def visit_Nil(self): return 'False'

    def visit_Num(self, n): return f'{n}'

    def visit_Z(self, i): return f'z{i}'

    def visit_Not(self, b): return f'(not {b.accept(self)})'

    def visit_Plus(self, x, y): return f'(+ {x.accept(self)} {y.accept(self)})'

    def visit_Minus(self, x, y): return f'(- {x.accept(self)} {y.accept(self)})'

    def visit_Times(self, x, y): return f'(* {x.accept(self)} {y.accept(self)})'

    def visit_Lt(self, x, y): return f'(< {x.accept(self)} {y.accept(self)})'

    def visit_And(self, x, y): return f'(and {x.accept(self)} {y.accept(self)})'

    def visit_If(self, b, x, y): return f'(if {b.accept(self)} {x.accept(self)} {y.accept(self)})'

    def visit_Point(self, x, y, color): return f'(P[{color}] {x.accept(self)} {y.accept(self)})'

    def visit_Line(self, x1, y1, x2, y2, color):
        return f'(L[{color.accept(self)}] {x1.accept(self)} {y1.accept(self)} {x2.accept(self)} {y2.accept(self)})'

    def visit_Rect(self, x1, y1, x2, y2, color):
        return f'(R[{color.accept(self)}] {x1.accept(self)} {y1.accept(self)} {x2.accept(self)} {y2.accept(self)})'

    def visit_Sprite(self, i, x, y):
        return f'(S{i} {x.accept(self)} {y.accept(self)})'

    def visit_Seq(self, bmps): return '[' + ' '.join([bmp.accept(self) for bmp in bmps]) + ' ]'

    def visit_Join(self, bmp1, bmp2): return f'[{bmp1.accept(self)} {bmp2.accept(self)}]'

    def visit_Intersect(self, bmp): return f'(intersect {bmp.accept(self)})'

    def visit_HFlip(self): return 'h-flip'

    def visit_VFlip(self): return 'v-flip'

    def visit_Translate(self, x, y): return f'(translate {x.accept(self)} {y.accept(self)})'

    def visit_Recolor(self, c): return f'recolor[{c.accept(self)}]'

    def visit_Compose(self, f, g): return f'(compose {f.accept(self)} {g.accept(self)})'
    
    def visit_Apply(self, f, bmp): return f'({f.accept(self)} {bmp.accept(self)})'

    def visit_Repeat(self, f, n): return f'(repeat {f.accept(self)} {n.accept(self)})'


def deserialize(seq):
    '''
    Deserialize a serialized seq into a program.
    '''
    def D(seq):
        if not seq: return []
        h, t = seq[0], D(seq[1:])
        if h == '': 
            return [Empty()] + t
        if isinstance(h, bool) and h == False: 
            return [Nil()] + t
        if isinstance(h, int): 
            return [Num(h)] + t
        if isinstance(h, str):
            if h.startswith('z_'): 
                return [Z(int(h[2:]))] + t
            if h.startswith('S_'):
                return [Sprite(int(h[2:]), t[0], t[1])] + t[2:]
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
        if h == 'L':
            return [Line(t[1], t[2], t[3], t[4], color=t[0])] + t[5:]
        if h == 'R':
            return [Rect(t[1], t[2], t[3], t[4], color=t[0])] + t[5:]
        if h =='H':
            return [HFlip()] + t
        if h =='V':
            return [VFlip()] + t
        if h =='T':
            return [Translate(t[0], t[1])] + t[2:]
        if h == 'RC':
            return [Recolor(t[0])] + t[1:]
        if h == 'C':
            return [Compose(t[0], t[1])] + t[2:]
        if h =='A':
            return [Apply(t[0], t[1])] + t[2:]
        if h =='R':
            return [Apply(t[0], t[1])] + t[2:]
        if h == '{':
            i = t.index('}')
            print(t[:i])
            return [Seq(*t[:i])] + t[i+1:]
        else:
            return seq

    tokens = D(seq)[0]
    return ' '.join(tokens)


class Serialize(Visitor):
    def __init__(self): pass

    def visit_Empty(self): return ['']

    def visit_Nil(self): return [False]

    def visit_Num(self, n): return [n]

    def visit_Z(self, i): return [f'z_{i}']

    def visit_Not(self, b): return ['~'] + b.accept(self)

    def visit_Plus(self, x, y): return ['+'] + x.accept(self) + y.accept(self)

    def visit_Minus(self, x, y): return ['-'] + x.accept(self) + y.accept(self)

    def visit_Times(self, x, y): return ['*'] + x.accept(self) + y.accept(self)

    def visit_Lt(self, x, y): return ['<'] + x.accept(self) + y.accept(self)

    def visit_And(self, x, y): return ['&'] + x.accept(self) + y.accept(self)

    def visit_If(self, b, x, y): return ['?'] + b.accept(self) + x.accept(self) + y.accept(self)

    def visit_Point(self, x, y, color): return ['P', color] + x.accept(self) + y.accept(self)

    def visit_Line(self, x1, y1, x2, y2, color):
        return ['L'] + color.accept(self) + x1.accept(self) + y1.accept(self) + x2.accept(self) + y2.accept(self)

    def visit_Rect(self, x1, y1, x2, y2, color):
        return ['R'] + color.accept(self) + x1.accept(self) + y1.accept(self) + x2.accept(self) + y2.accept(self)

    def visit_Sprite(self, i, x, y):
        return [f'S_{i}'] + x.accept(self) + y.accept(self)

    def visit_Seq(self, bmps): 
        l = ['{'];
        for bmp in bmps:
            l.extend(bmp.accept(self))
        l.append('}')           # stop symbol
        return l

    def visit_Join(self, bmp1, bmp2): return [';'] + bmp1.accept(self) + bmp2.accept(self)

    def visit_Intersect(self, bmp): return ['I'] + bmp.accept(self)

    def visit_HFlip(self): return ['H']

    def visit_VFlip(self): return ['V']

    def visit_Translate(self, x, y): return ['T'] + x.accept(self) + y.accept(self)

    def visit_Recolor(self, c): return ['RC'] + c.accept(self)

    def visit_Compose(self, f, g): return ['C'] + f.accept(self) + g.accept(self)
    
    def visit_Apply(self, f, bmp): return ['A'] + f.accept(self) + bmp.accept(self)

    def visit_Repeat(self, f, n): return ['R'] + f.accept(self) + n.accept(self)


class Zs(Visitor):
    def __init__(self): pass

    def visit_Empty(self): return set()

    def visit_Nil(self): return set()

    def visit_Num(self, n): return set()

    def visit_Z(self, i): return {i}

    def visit_Not(self, b): return b.accept(self)

    def visit_Plus(self, x, y): return x.accept(self) | y.accept(self)

    def visit_Minus(self, x, y): return x.accept(self) | y.accept(self)

    def visit_Times(self, x, y): return x.accept(self) | y.accept(self)

    def visit_Lt(self, x, y): return x.accept(self) | y.accept(self)

    def visit_And(self, x, y): return x.accept(self) | y.accept(self)

    def visit_If(self, b, x, y): return b.accept(self) | x.accept(self) | y.accept(self)

    def visit_Point(self, x, y, color): return x.accept(self) | y.accept(self) | color.accept(self)

    def visit_Line(self, x1, y1, x2, y2, color):
        return x1.accept(self) | y1.accept(self) | x2.accept(self) | y2.accept(self) | color.accept(self)

    def visit_Rect(self, x1, y1, x2, y2, color):
        return x1.accept(self) | y1.accept(self) | x2.accept(self) | y2.accept(self) | color.accept(self)

    def visit_Sprite(self, i, x, y):
        return x.accept(self) | y.accept(self)

    def visit_Seq(self, bmps): return set.union(bmp.accept(self) for bmp in bmps)

    def visit_Join(self, bmp1, bmp2): return bmp1.accept(self) | bmp2.accept(self)

    def visit_Intersect(self, bmp): return bmp.accept(self)

    def visit_HFlip(self): return set()

    def visit_VFlip(self): return set()

    def visit_Translate(self, x, y): return x.accept(self) | y.accept(self)

    def visit_Recolor(self, c): return c.accept(self)

    def visit_Compose(self, f, g): return f.accept(self) | g.accept(self)

    def visit_Apply(self, f, bmp): return f.accept(self) | bmp.accept(self)

    def visit_Repeat(self, f, n): return f.accept(self) | n.accept(self)


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
        (Rect(Num(0), Num(0),
              Num(1), Num(2)),
         lambda z: util.img_to_tensor(["##__",
                                       "##__",
                                       "##__",
                                       "____"], w=B_W, h=B_H)),
    ]
    for expr, correct_semantics in tests:
        for x in range(10):
            for y in range(10):
                out = expr.eval({"z": [x, y]})
                expected = correct_semantics([x, y])
                t = expr.out_type
                # print(expr, out, expected)
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
    expr = Rect(Num(0), Num(0),
                Num(1), Num(1))
    out = expr.eval({'z': []})
    expected = util.img_to_tensor(["##__",
                                   "##__",
                                   "____",
                                   "____"], w=B_W, h=B_H)
    assert T.equal(expected, out), f"test_render failed:\n expected={expected},\n out={out}"

    # (1,0), (3,3)
    expr = Rect(Z(0),
                Num(0),
                Plus(Num(2), Num(1)),
                Num(3))
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
        (Line(Num(0), Num(0),
              Num(1), Num(1)),
         ["#___",
          "_#__",
          "____",
          "____"]),
        (Line(Num(0), Num(0),
              Num(3), Num(3)),
         ["#___",
          "_#__",
          "__#_",
          "___#"]),
        (Line(Num(1), Num(0),
              Num(3), Num(2)),
         ["_#__",
          "__#_",
          "___#",
          "____"]),
        (Line(Num(1), Num(2),
              Num(2), Num(3)),
         ["____",
          "____",
          "_#__",
          "__#_"]),
        (Line(Num(1), Num(0),
              Num(3), Num(0)),
         ["_###",
          "____",
          "____",
          "____"]),
        (Line(Num(1), Num(2),
              Num(1), Num(3)),
         ["____",
          "____",
          "_#__",
          "_#__"]),

        # Reflection
        (Apply(HFlip(), 
               Line(Num(0), Num(0),
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
        (Join(Rect(Num(0), Num(0),
                   Num(1), Num(1)),
              Line(Num(2), Num(3),
                   Num(3), Num(3))),
         ["##__",
          "##__",
          "____",
          "__##"]),
        (Apply(HFlip(),
               Join(Rect(Num(0), Num(0),
                         Num(1), Num(1)),
                    Rect(Num(2), Num(2),
                         Num(3), Num(3)))),
         ["##__" + "_"*(B_W-8) + "__##",
          "##__" + "_"*(B_W-8) + "__##",
          "__##" + "_"*(B_W-8) + "##__",
          "__##" + "_"*(B_W-8) + "##__"]),

        # Translate
        (Apply(Translate(Num(0), Num(0)),
               Line(Num(0), Num(0),
                    Num(3), Num(3))),
         ["#___",
          "_#__",
          "__#_",
          "___#"]),
        (Apply(Compose(Translate(Num(1), Num(0)), Recolor(Num(2))),
               Line(Num(0), Num(0),
                    Num(3), Num(3))),
         ["12___",
          "_12__",
          "__12_",
          "___12"]),
        (Apply(Compose(Translate(Num(-1), Num(0)), Recolor(Num(2))),
               Line(Num(0), Num(0),
                    Num(3), Num(3))),
         ["1____",
          "21___",
          "_21__",
          "__21_"]),
        (Apply(Compose(Translate(Num(0), Num(1)), Recolor(Num(2))),
               Line(Num(0), Num(0),
                    Num(3), Num(3))),
         ["1___",
          "21__",
          "_21_",
          "__21",
          "___2"]),
        (Apply(Compose(Translate(Num(0), Num(-1)), Recolor(Num(2))),
               Line(Num(0), Num(0),
                    Num(3), Num(3))),
         ["12___",
          "_12__",
          "__12_",
          "___1_"]),
        (Apply(Compose(Translate(Num(-1), Num(-1)), Recolor(Num(2))),
               Line(Num(0), Num(0),
                    Num(3), Num(3))),
         ["2___",
          "_2__",
          "__2_",
          "___1"]),
        (Apply(Compose(Translate(Num(1), Num(1)), Recolor(Num(2))),
               Line(Num(0), Num(0),
                    Num(3), Num(3))),
         ["1____",
          "_2___",
          "__2__",
          "___2_",
          "____2"]),
        (Apply(Compose(Translate(Num(2), Num(3)), Recolor(Num(2))),
               Line(Num(0), Num(0),
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
               Line(Num(0), Num(0),
                    Num(3), Num(3))),
         ["#_#_#___",
          "_#_#_#__",
          "__#_#_#_",
          "___#_#_#"]),
        (Apply(Repeat(Compose(Translate(Num(2), Num(0)),
                              Recolor(Num(2))),
                      Num(2)),
               Line(Num(0), Num(0),
                    Num(3), Num(3))),
         ["1_2_2___",
          "_1_2_2__",
          "__1_2_2_",
          "___1_2_2"]),
        (Apply(Intersect(Rect(Num(0), Num(0), Num(3), Num(3))),
               Line(Num(0), Num(0), Num(3), Num(3))),
         ["#___",
          "_#__",
          "__#_",
          "___#"]),
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
        

def test_sprite():
    tests = [
        ([["2___",
           "2___",
           "____",
           "____"],
        ],
         Sprite(0),
         ["2___",
          "2___",
          "____",
          "____"]),
        ([
            ["__",
             "__"],
            ["11",
             "_1"],
        ],
         Sprite(0),
         ["__",
          "__"]),
        ([
            ["__",
             "__"],
            ["11",
             "_1"],
        ],
         Sprite(1),
         ["11",
          "_1"]),
        ([
            ["111",
             "__1",
             "_1_"]
        ],
         Apply(Compose(HFlip(),
                       Recolor(Num(2))),
               Sprite(0)),
            ["111" + '_' * (B_W - 6) + "222",
             "__1" + '_' * (B_W - 6) + "2__",
             "_1_" + '_' * (B_W - 6) + "_2_"]),
        ([
            ["111",
             "__1",
             "_1_"]
        ],
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

def test_eval_color():
    tests = [
        (Rect(Num(0), Num(0),
              Num(1), Num(1), Num(2)),
         ["22__",
          "22__",
          "____",
          "____"]),
        (Line(Num(1), Num(0),
              Num(3), Num(2), Num(3)),
         ["_3__",
          "__3_",
          "___3",
          "____"]),
        (Line(Num(1), Num(0),
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
        (Join(Rect(Num(0), Num(0),
                   Num(1), Num(1), Num(1)),
              Rect(Num(2), Num(2),
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
        (Rect(Num(0), Num(1), Num(3), Num(3)),
         set()),
        (Rect(Z(0), Z(1), Num(3), Num(3)),
         {0, 1}),
        (Rect(Z(0), Z(1), Z(2), Z(3)),
         {0, 1, 2, 3}),
        (Rect(Z(0), Z(1), Z(0), Z(1)),
         {0, 1}),
    ]
    for expr, ans in test_cases:
        out = expr.zs()
        assert out == ans, f"test_zs failed: expected={ans}, actual={out}"
    print(" [+] passed test_zs")


if __name__ == '__main__':
    test_eval()
    test_eval_bitmap()
    test_eval_color()
    test_sprite()
    # test_zs()
