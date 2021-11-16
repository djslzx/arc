from bmap import Bitmap
import torch as T

# bitmap size constants
B_W=4
B_H=4

# constants for z_n, z_b
Z_SIZE = 6                        # length of z_n, z_b 
Z_LO = 0                          # min poss value in z_n
Z_HI = max(B_W, B_H) # max poss value in z_n

class Grammar:
    def __init__(self, ops, consts):
        self.ops = ops
        self.consts = consts
    
class Expr:
    def eval(self, environment):
        assert False, f"not implemented for {self}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self): return hash(str(self))

    def __ne__(self, other): return str(self) != str(other)

    def __gt__(self, other): return str(self) > str(other)

    def __lt__(self, other): return str(self) < str(other)

    def dist(self, other):
        # TODO
        # Levenshtein distance? but should also account for differences between scalars
        assert False, "unimplemented"

    def zs(self):
        return set()

class FALSE(Expr):
    argument_types = []
    return_type = "bool"
    
    def __init__(self): pass

    def __str__(self):
        return "False"

    def pretty_print(self):
        return "False"

    def eval(self, env):
        return False

class Num(Expr):
    argument_types = []
    return_type = "int"
    
    def __init__(self, n):
        self.n = n

    def __str__(self):
        return f"Num({self.n})"

    def pretty_print(self):
        return str(self.n)

    def eval(self, env):
        return self.n
    
class Zb(Expr):
    argument_types = ["int"]
    return_type = "bool"
    
    def __init__(self, i):
        self.i = i

    def __str__(self):
        return f"Zb('{self.i}')"

    def pretty_print(self):
        try:
            return f"z_b[{self.i.pretty_print()}]"
        except AttributeError:
            return f"z_b[{self.i}]"

    def eval(self, env):
        i = self.i.eval(env)
        assert isinstance(i, int)
        return env["z_b"][i]

    def zs(self):
        return {self.i.eval({})}

class Zn(Expr):
    """z_n"""
    argument_types = ["int"]
    return_type = "int"
    
    def __init__(self, i):
        self.i = i

    def __str__(self):
        return f"Zn('{self.i}')"

    def pretty_print(self):
        try:
            return f"z_n[{self.i.pretty_print()}]"
        except AttributeError:
            return f"z_n[{self.i}]"

    def eval(self, env):
        i = self.i.eval(env)
        assert isinstance(i, int)
        out = env["z_n"][i]
        if isinstance(out, T.LongTensor):
            return out.item()
        else:
            return out

    def zs(self):
        return {self.i.eval({})}

class Plus(Expr):
    argument_types = ["int","int"]
    return_type = "int"
    
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Plus({self.x}, {self.y})"

    def pretty_print(self):
        return f"(+ {self.x.pretty_print()} {self.y.pretty_print()})"

    def eval(self, env):
        x = self.x.eval(env)
        y = self.y.eval(env)
        assert isinstance(x, int) and isinstance(y, int)
        return x + y

    def zs(self):
        return set.union(self.x.zs(), self.y.zs())

class Minus(Expr):
    argument_types = ["int","int"]
    return_type = "int"
    
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Minus({self.x}, {self.y})"

    def pretty_print(self):
        return f"(- {self.x.pretty_print()} {self.y.pretty_print()})"

    def eval(self, env):
        x = self.x.eval(env)
        y = self.y.eval(env)
        assert isinstance(x, int) and isinstance(y, int)
        return x - y

    def zs(self):
        return set.union(self.x.zs(), self.y.zs())

class Times(Expr):
    argument_types = ["int","int"]
    return_type = "int"
    
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Times({self.x}, {self.y})"

    def pretty_print(self):
        return f"(* {self.x.pretty_print()} {self.y.pretty_print()})"

    def eval(self, env):
        x = self.x.eval(env)
        y = self.y.eval(env)
        assert isinstance(x, int) and isinstance(y, int)
        return x * y

    def zs(self):
        return set.union(self.x.zs(), self.y.zs())

class Lt(Expr):
    argument_types = ["int","int"]
    return_type = "bool"
    
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Lt({self.x}, {self.y})"

    def pretty_print(self):
        return f"(< {self.x.pretty_print()} {self.y.pretty_print()})"

    def eval(self, env):
        x = self.x.eval(env)
        y = self.y.eval(env)
        assert isinstance(x, int) and isinstance(y, int)
        return x < y

    def zs(self):
        return set.union(self.x.zs(), self.y.zs())

class And(Expr):
    argument_types = ["bool","bool"]
    return_type = "bool"
    
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"And({self.x}, {self.y})"

    def pretty_print(self):
        return f"(and {self.x.pretty_print()} {self.y.pretty_print()})"

    def eval(self, env):
        x = self.x.eval(env)
        y = self.y.eval(env)
        assert isinstance(x, bool) and isinstance(y, bool)
        return x and y

    def zs(self):
        return set.union(self.x.zs(), self.y.zs())

class Not(Expr):
    argument_types = ["bool"]
    return_type = "bool"
    
    def __init__(self, x):
        self.x = x

    def __str__(self):
        return f"Not({self.x})"

    def pretty_print(self):
        return f"(not {self.x.pretty_print()})"

    def eval(self, env):
        x = self.x.eval(env)
        assert isinstance(x, bool)
        return not x

    def zs(self):
        return self.x.zs()

class If(Expr):
    argument_types = ["bool","int","int"]
    return_type = "Bitmap"
    
    def __init__(self, test, yes, no):
        self.test, self.yes, self.no = test, yes, no

    def __str__(self):
        return f"If({self.test}, {self.yes}, {self.no})"

    def pretty_print(self):
        return f"(if {self.test.pretty_print()} {self.yes.pretty_print()} {self.no.pretty_print()})"

    def eval(self, env):
        test = self.test.eval(env)
        yes = self.yes.eval(env)
        no = self.no.eval(env)
        assert isinstance(test, bool) and \
            isinstance(yes, int) and \
            isinstance(no, int), \
            f"[If] type mismatch: t(test)={type(test)}, t(yes)={type(yes)}, t(no)={type(no)}"
        return yes if test else no

    def zs(self):
        return set.union(self.test.zs(),
                         self.yes.zs(),
                         self.no.zs())

class Rect(Expr):
    """
    A rectangle Rect(Point(a,b), Point(c,d)) generates a bitmap st points (x,y) are set to 1, 
    where
      a <= x < c, b <= y < d.

    E.g. A 1x1 rectangle at the origin is Rect(Point(0,0), Point(1,1))

    Invariants:
    - Rectangles may not be empty
    - Rectangles must stay in the bitmap (W x H)
    - The first point should be the bottom left corner and the second point should be the top right corner.
    - So, for Rect(a,b,c,d), 0 <= a < c <= W and 0 <= b < d <= H
    """

    argument_types = ["int", "int", "int", "int"]
    return_type = "Bitmap"
    
    def __init__(self, x1, y1, x2, y2):
        # assert isinstance(p1, Point) and isinstance(p2, Point)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        
    def __str__(self):
        return f"Rect({self.x1}, {self.y1}, {self.x2}, {self.y2})"

    def pretty_print(self):
        return f"R({self.x1.pretty_print()}, {self.y1.pretty_print()}, {self.x2.pretty_print()}, {self.y2.pretty_print()})"

    def eval(self, env):
        x1 = self.x1.eval(env)
        y1 = self.y1.eval(env)
        x2 = self.x2.eval(env)
        y2 = self.y2.eval(env)
        assert 0 <= x1 < x2 <= B_W and 0 <= y1 < y2 <= B_H
        return Bitmap([[x1 <= x < x2 and y1 <= y < y2
                        for x in range(B_W)]
                       for y in range(B_H)])

    def zs(self):
        return set.union(self.x1.zs(),
                         self.y1.zs(),
                         self.x2.zs(),
                         self.y2.zs())

class Program(Expr):
    argument_types = ["Bitmap", "Bitmap"]
    return_type = "Bitmap"
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __str__(self):
        return f"Program({self.x}, {self.y})"

    def pretty_print(self):
        return (f"{self.x.pretty_print()}; {self.y.pretty_print()}")

    def eval(self, env):
        x = self.x.eval(env)
        y = self.y.eval(env)
        return Bitmap.union(x, y)

    def zs(self):
        return set.union(self.x.zs(), self.y.zs())

def test_eval():
    tests = [
        (FALSE(),
         lambda zb,zn: False),
        (Not(FALSE()),
         lambda zb, zn: True),
        (Times(Zn(Num(0)), Zn(Num(1))),
         lambda zb, zn: zn[0] * zn[1]),
        (If(Lt(Zn(Num(0)), Zn(Num(1))),
            Zn(Num(0)), 
            Zn(Num(1))),
         lambda zb, zn: min(zn[0], zn[1])),
        (If(Not(Lt(Zn(Num(0)), 
                   Zn(Num(1)))),
            Times(Zn(Num(0)), Zn(Num(1))), 
            Plus(Zn(Num(0)), Zn(Num(1)))), 
         lambda zb, zn: zn[0] * zn[1] if not (zn[0] < zn[1]) else zn[0] + zn[1]),
        (Rect(Num(0),
              Num(0), 
              Num(1),
              Num(2)),
         lambda zb, zn: Bitmap.from_img(["#___",
                                         "#___",
                                         "____",
                                         "____",])),
        (Program(Rect(Num(0),
                      Num(0), 
                      Num(1),
                      Num(1)),
                 Rect(Num(2),
                      Num(3), 
                      Num(4),
                      Num(4))),
         lambda zb, zn: Bitmap.from_img(["#___",
                                         "____",
                                         "____",
                                         "__##"]))
    ]
    for expr, correct_semantics in tests:
        for x in range(10):
            for y in range(10):
                out = expr.eval({"z_b":[], "z_n":[x,y]})
                expected = correct_semantics([], [x,y])
                assert out == expected, f"failed eval test:\n"\
                    f" expected=\n{expected.pretty_print()}\n"\
                    f" out=\n{out.pretty_print()}"
    print(" [+] eval passes checks")

def test_render():
    # (0,0), (1,1)
    expr = Rect(Num(0),Num(0), 
                Num(1),Num(1))
    out = expr.eval({})
    expected = Bitmap.from_img(["#___",
                                "____",
                                "____",
                                "____"])
    assert expected == out, f"test_render failed:\n expected={expected},\n out={out}"

    # (1,0), (3,3)
    expr = Rect(Zn(Num(0)), 
                Num(0), 
                Plus(Num(2), Num(1)), 
                Num(3)) 
    out = expr.eval({'z_n': [1,2,3]})
    expected = Bitmap.from_img(["_##_",
                                "_##_",
                                "_##_",
                                "____"])
    assert expected == out, f"test_render failed:\n expected={expected},\n out={out}"
    print(" [+] passed test_render")

def test_zs():
    test_cases = [
        (Rect(Num(0), Num(1), Num(4), Num(4)), 
         set()),
        (Rect(Zn(Num(0)), Zn(Num(1)), Num(4), Num(4)), 
         {0, 1}),
        (Rect(Zn(Num(0)), Zn(Num(1)), Zn(Num(2)), Zn(Num(3))),
         {0, 1, 2, 3}),
        (Rect(Zn(Num(0)), Zn(Num(1)), Zn(Num(0)), Zn(Num(1))),
         {0, 1}),
    ]
    for expr, ans in test_cases:
        out = expr.zs()
        assert out == ans, f"test_zs failed: expected={ans}, actual={out}"

    print(" [+] passed test_zs")

if __name__ == '__main__':
    test_eval()
    test_render()
    test_zs()
