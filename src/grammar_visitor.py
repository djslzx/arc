import util
import torch as T

# bitmap size constants
B_W = 8
B_H = 8

# constants for z_n, z_b
Z_SIZE = 6  # length of z_n, z_b
Z_LO = 0  # min poss value in z_n
Z_HI = max(B_W, B_H)  # max poss value in z_n

'''
Grammar

- Expr: F, Num, Z, +, -, *, <, and, not, if, rect, prog
+ reflect, diagonal/horizontal/vertical line
'''


class Grammar:
    def __init__(self, ops, consts):
        self.ops = ops
        self.consts = consts


class Visited:
    def accept(self, visitor):
        assert False, "Visited subclass should implement `accept`"

    def eval(self, env): return self.accept(Eval(env))

    def zs(self): return self.accept(Zs())

    def __str__(self): return self.accept(Print())


class Expr(Visited):
    def accept(self, visitor): assert False, f"not implemented for {self}"

    def __repr__(self): return str(self)

    def __eq__(self, other): return str(self) == str(other)

    def __hash__(self): return hash(str(self))

    def __ne__(self, other): return str(self) != str(other)

    def __gt__(self, other): return str(self) > str(other)

    def __lt__(self, other): return str(self) < str(other)


class F(Expr):
    in_types = []
    out_type = 'bool'

    def __init__(self): pass

    def accept(self, v): return v.visit_F()


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
    in_types = ['int', 'int', 'int', 'int']
    out_type = 'bitmap'

    def __init__(self, x1, y1, x2, y2, color=1):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color

    def accept(self, v): return v.visit_Line(self.x1, self.y1, self.x2, self.y2, self.color)


class Point(Expr):
    in_types = ['int', 'int']
    out_type = 'bitmap'

    def __init__(self, x, y, color=1):
        self.x = x
        self.y = y
        self.color = color

    def accept(self, v): return v.visit_Point(self.x, self.y, self.color)


class Rect(Expr):
    in_types = ['int', 'int', 'int', 'int']
    out_type = 'bitmap'

    def __init__(self, x1, y1, x2, y2, color=1):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color

    def accept(self, v): return v.visit_Rect(self.x1, self.y1, self.x2, self.y2, self.color)


class Stack(Expr):
    in_types = ['bitmap', 'bitmap']
    out_type = 'bitmap'

    def __init__(self, b1, b2):
        self.b1 = b1
        self.b2 = b2

    def accept(self, v): return v.visit_Stack(self.b1, self.b2)


class Intersect(Expr):
    in_types = ['bitmap', 'bitmap']
    out_type = 'bitmap'

    def __init__(self, b1, b2):
        self.b1 = b1
        self.b2 = b2

    def accept(self, v): return v.visit_Intersect(self.b1, self.b2)


class ReflectH(Expr):
    in_types = ['bitmap']
    out_type = 'bitmap'

    def __init__(self, b): self.b = b

    def accept(self, v): return v.visit_ReflectH(self.b)


class ReflectV(Expr):
    in_types = ['bitmap']
    out_type = 'bitmap'

    def __init__(self, b): self.b = b

    def accept(self, v): return v.visit_ReflectV(self.b)


class Visitor:
    @staticmethod
    def fail(s): assert False, f"Visitor subclass should implement `{s}`"

    def visit_F(self): self.fail('F')

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

    def visit_Stack(self, b1, b2): self.fail('Union')

    def visit_Intersect(self, b1, b2): self.fail('Intersect')

    def visit_ReflectH(self, b): self.fail('ReflectH')


class Eval(Visitor):
    def __init__(self, env):
        self.env = env

    @classmethod
    def make_bitmap(cls, f):
        return T.tensor([[f((x, y))
                          for x in range(B_W)]
                         for y in range(B_H)]).float()

    def visit_F(self):
        return False

    def visit_Num(self, n):
        return n

    def visit_Z(self, i):
        assert 'z' in self.env, "Eval env missing Z"
        z = self.env['z'][i]
        return z.item() if isinstance(z, T.LongTensor) else z

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
        x, y = x.accept(self), y.accept(self)
        assert isinstance(x, int) and isinstance(y, int)
        return Eval.make_bitmap(lambda p: (p[0] == x and p[1] == y) * color)

    def visit_Line(self, x1, y1, x2, y2, color):
        x1, y1, x2, y2 = (x1.accept(self), y1.accept(self),
                          x2.accept(self), y2.accept(self))
        assert all(isinstance(v, int) for v in [x1, y1, x2, y2])
        assert 0 <= x1 <= x2 <= B_W and 0 <= y1 <= y2 <= B_H
        assert x1 == x2 or y1 == y2 or abs(x2 - x1) == abs(y2 - y1)
        if x1 == x2:  # vertical
            return Eval.make_bitmap(lambda p: (x1 == p[0] and y1 <= p[1] <= y2) * color)
        elif y1 == y2:  # horizontal
            return Eval.make_bitmap(lambda p: (x1 <= p[0] <= x2 and y1 == p[1]) * color)
        else:  # diagonal
            return Eval.make_bitmap(lambda p: (x1 <= p[0] <= x2 and
                                               y1 <= p[1] <= y2 and
                                               p[1] == y1 + (p[0] - x1)) * color)

    def visit_Rect(self, x1, y1, x2, y2, color):
        x1, y1, x2, y2 = (x1.accept(self), y1.accept(self),
                          x2.accept(self), y2.accept(self))
        assert all(isinstance(v, int) for v in [x1, y1, x2, y2])
        assert 0 <= x1 < x2 <= B_W and 0 <= y1 < y2 <= B_H
        return Eval.make_bitmap(lambda p: (x1 <= p[0] < x2 and y1 <= p[1] < y2) * color)

    def visit_Stack(self, b1, b2):
        b1, b2 = b1.accept(self), b2.accept(self)
        assert isinstance(b1, T.FloatTensor) and isinstance(b2, T.FloatTensor), \
            f"Union needs two float tensors, found b1={b1}, b2={b2}"
        def f(p):
            x, y = p
            c1, c2 = b1[y][x], b2[y][x]
            return c1 if c1 > 0 else c2
        return Eval.make_bitmap(f)

    def visit_Intersect(self, b1, b2):
        b1, b2 = b1.accept(self), b2.accept(self)
        assert isinstance(b1, T.FloatTensor) and isinstance(b2, T.FloatTensor), \
            f"Intersect needs two float tensors, found b1={b1}, b2={b2}"
        def f(p):
            x, y = p
            c1, c2 = b1[y][x], b2[y][x]
            if c1 == 0: return c2
            elif c2 == 0: return c1
            else: return 0
        return Eval.make_bitmap(f)

    def visit_ReflectH(self, b):
        b = b.accept(self)
        assert isinstance(b, T.FloatTensor)
        return b.flip(1)

    def visit_ReflectV(self, b):
        b = b.accept(self)
        assert isinstance(b, T.FloatTensor)
        return b.flip(0)


class Print(Visitor):
    def __init__(self): pass

    def visit_F(self): return 'False'

    def visit_Num(self, n): return f'{n}'

    def visit_Z(self, i): return f'z[{i}]'

    def visit_Not(self, b): return f'(not {b.accept(self)})'

    def visit_Plus(self, x, y): return f'(+ {x.accept(self)} {y.accept(self)})'

    def visit_Minus(self, x, y): return f'(- {x.accept(self)} {y.accept(self)})'

    def visit_Times(self, x, y): return f'(* {x.accept(self)} {y.accept(self)})'

    def visit_Lt(self, x, y): return f'(< {x.accept(self)} {y.accept(self)})'

    def visit_And(self, x, y): return f'(and {x.accept(self)} {y.accept(self)})'

    def visit_If(self, b, x, y): return f'(if {b} {x.accept(self)} {y.accept(self)})'

    def visit_Point(self, x, y, color): return f'(P[{color}] {x.accept(self)} {y.accept(self)})'

    def visit_Line(self, x1, y1, x2, y2, color):
        return f'(L[{color}] {x1.accept(self)} {y1.accept(self)} {x2.accept(self)} {y2.accept(self)})'

    def visit_Rect(self, x1, y1, x2, y2, color):
        return f'(R[{color}] {x1.accept(self)} {y1.accept(self)} {x2.accept(self)} {y2.accept(self)})'

    def visit_Stack(self, b1, b2): return f'(u {b1.accept(self)} {b2.accept(self)})'

    def visit_Intersect(self, b1, b2): return f'(n {b1.accept(self)} {b2.accept(self)})'

    def visit_ReflectH(self, b): return f'(reflect-h {b.accept(self)})'

    def visit_ReflectV(self, b): return f'(reflect-v {b.accept(self)})'


class Zs(Visitor):
    def __init__(self): pass

    def visit_F(self): return set()

    def visit_Num(self, n): return set()

    def visit_Z(self, i): return {i}

    def visit_Not(self, b): return b.accept(self)

    def visit_Plus(self, x, y): return x.accept(self) | y.accept(self)

    def visit_Minus(self, x, y): return x.accept(self) | y.accept(self)

    def visit_Times(self, x, y): return x.accept(self) | y.accept(self)

    def visit_Lt(self, x, y): return x.accept(self) | y.accept(self)

    def visit_And(self, x, y): return x.accept(self) | y.accept(self)

    def visit_If(self, b, x, y): return b.accept(self) | x.accept(self) | y.accept(self)

    def visit_Point(self, x, y, color): return x.accept(self) | y.accept(self)

    def visit_Rect(self, x1, y1, x2, y2, color):
        return x1.accept(self) | y1.accept(self) | x2.accept(self) | y2.accept(self)

    def visit_Line(self, x1, y1, x2, y2, color):
        return x1.accept(self) | y1.accept(self) | x2.accept(self) | y2.accept(self)

    def visit_Stack(self, b1, b2): return b1.accept(self) | b2.accept(self)

    def visit_Intersect(self, b1, b2): return b1.accept(self) | b2.accept(self)

    def visit_ReflectH(self, b): return b.accept(self)

    def visit_ReflectV(self, b): return b.accept(self)


def test_eval():
    tests = [
        (F(),
         lambda z: False),
        (Not(F()),
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
         lambda z: util.img_to_tensor(["#___",
                                       "#___",
                                       "____",
                                       "____"], w=B_W, h=B_H)),
        (Line(Num(0), Num(0),
              Num(1), Num(1)),
         lambda z: util.img_to_tensor(["#___",
                                       "_#__",
                                       "____",
                                       "____"], w=B_W, h=B_H)),
        (Line(Num(0), Num(0),
              Num(3), Num(3)),
         lambda z: util.img_to_tensor(["#___",
                                       "_#__",
                                       "__#_",
                                       "___#"], w=B_W, h=B_H)),
        (Line(Num(1), Num(0),
              Num(3), Num(2)),
         lambda z: util.img_to_tensor(["_#__",
                                       "__#_",
                                       "___#",
                                       "____"], w=B_W, h=B_H)),
        (Line(Num(1), Num(2),
              Num(2), Num(3)),
         lambda z: util.img_to_tensor(["____",
                                       "____",
                                       "_#__",
                                       "__#_"], w=B_W, h=B_H)),
        (Line(Num(1), Num(0),
              Num(3), Num(0)),
         lambda z: util.img_to_tensor(["_###",
                                       "____",
                                       "____",
                                       "____"], w=B_W, h=B_H)),
        (Line(Num(1), Num(2),
              Num(1), Num(3)),
         lambda z: util.img_to_tensor(["____",
                                       "____",
                                       "_#__",
                                       "_#__"], w=B_W, h=B_H)),
        (ReflectH(Line(Num(0), Num(0),
                       Num(3), Num(3))),
         lambda z: util.img_to_tensor(["_"*(B_W-4) + "___#",
                                       "_"*(B_W-4) + "__#_",
                                       "_"*(B_W-4) + "_#__",
                                       "_"*(B_W-4) + "#___"], w=B_W, h=B_H)),
        (Stack(Stack(Point(Num(0), Num(0)),
                     Point(Num(1), Num(3))),
               Stack(Point(Num(2), Num(0)),
                     Point(Num(3), Num(1)))),
         lambda z: util.img_to_tensor(["#_#_",
                                       "___#",
                                       "____",
                                       "_#__"], w=B_W, h=B_H)),
        (ReflectV(Stack(Stack(Point(Num(0), Num(0)),
                              Point(Num(1), Num(3))),
                        Stack(Point(Num(2), Num(0)),
                              Point(Num(3), Num(1))))),
         lambda z: util.img_to_tensor(["____"] * (B_H - 4) +
                                      ["_#__",
                                       "____",
                                       "___#",
                                       "#_#_"], w=B_W, h=B_H)),
        (Stack(Rect(Num(0), Num(0),
                    Num(1), Num(1)),
               Rect(Num(2), Num(3),
                    Num(4), Num(4))),
         lambda z: util.img_to_tensor(["#___",
                                       "____",
                                       "____",
                                       "__##"], w=B_W, h=B_H)),
        (ReflectH(Stack(Rect(Num(0), Num(0),
                             Num(1), Num(1)),
                        Rect(Num(2), Num(3),
                             Num(4), Num(4)))),
         lambda z: util.img_to_tensor(["_"*(B_W-4) + "___#",
                                       "_"*(B_W-4) + "____",
                                       "_"*(B_W-4) + "____",
                                       "_"*(B_W-4) + "##__"], w=B_W, h=B_H)),
        (ReflectH(ReflectH(Stack(Rect(Num(0), Num(0),
                                      Num(1), Num(1)),
                                 Rect(Num(2), Num(3),
                                      Num(4), Num(4))))),
         lambda z: util.img_to_tensor(["#___",
                                       "____",
                                       "____",
                                       "__##"], w=B_W, h=B_H)),
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
    expected = util.img_to_tensor(["#___",
                                   "____",
                                   "____",
                                   "____"], w=B_W, h=B_H)
    assert T.equal(expected, out), f"test_render failed:\n expected={expected},\n out={out}"

    # (1,0), (3,3)
    expr = Rect(Z(0),
                Num(0),
                Plus(Num(2), Num(1)),
                Num(3))
    out = expr.eval({'z': [1, 2, 3]})
    expected = util.img_to_tensor(["_##_",
                                   "_##_",
                                   "_##_",
                                   "____"], w=B_W, h=B_H)
    assert T.equal(expected, out), f"test_render failed:\n expected={expected},\n out={out}"
    print(" [+] passed test_eval")

def test_eval_color():
    tests = [
        (Rect(Num(0), Num(0),
              Num(1), Num(2), 2),
         lambda z: util.img_to_tensor(["2___",
                                       "2___",
                                       "____",
                                       "____"], w=B_W, h=B_H)),
        (Line(Num(1), Num(0),
              Num(3), Num(2), 3),
         lambda z: util.img_to_tensor(["_3__",
                                       "__3_",
                                       "___3",
                                       "____"], w=B_W, h=B_H)),
        (Line(Num(1), Num(0),
              Num(3), Num(0), 2),
         lambda z: util.img_to_tensor(["_222",
                                       "____",
                                       "____",
                                       "____"], w=B_W, h=B_H)),
        (Stack(Stack(Point(Num(0), Num(0), 2),
                     Point(Num(1), Num(3), 3)),
               Stack(Point(Num(2), Num(0), 7),
                     Point(Num(3), Num(1), 9))),
         lambda z: util.img_to_tensor(["2_7_",
                                       "___9",
                                       "____",
                                       "_3__"], w=B_W, h=B_H)),
        (Stack(Rect(Num(0), Num(0),
                    Num(1), Num(1), 1),
               Rect(Num(2), Num(2),
                    Num(4), Num(4), 6)),
         lambda z: util.img_to_tensor(["1___",
                                       "____",
                                       "__66",
                                       "__66"], w=B_W, h=B_H)),
    ]
    for expr, correct_semantics in tests:
        for x in range(10):
            for y in range(10):
                out = expr.eval({"z": [x, y]})
                expected = correct_semantics([x, y])
                t = expr.out_type
                assert T.equal(out, expected), \
                    f"failed eval test:\n" \
                    f" expr=\n{expr}\n" \
                    f" expected=\n{expected}\n" \
                    f" out=\n{out}"
    print(" [+] passed test_eval_color")

def test_zs():
    test_cases = [
        (Rect(Num(0), Num(1), Num(4), Num(4)),
         set()),
        (Rect(Z(0), Z(1), Num(4), Num(4)),
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
    test_eval_color()
    test_zs()
