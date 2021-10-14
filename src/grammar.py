from bmap import Bitmap

# bitmap size constants
BMP_WIDTH=4
BMP_HEIGHT=4

# constants for z_n, z_b
Z_SIZE = 16                     # length of z_n, z_b 
Z_LO = 0                        # min poss value in z_n
Z_HI = Z_SIZE                   # max poss value in z_n

class Expr():
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
    
class Zb(Expr):
    argument_types = ["int"]
    return_type = "bool"
    
    def __init__(self, i):
        self.i = i

    def __str__(self):
        return f"Zb('{self.i}')"

    def pretty_print(self):
        return f"z_b[{self.i}]"

    def eval(self, env):
        i = self.i.eval(env)
        assert isinstance(i, int)
        return env["z_b"][i]

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

class Zn(Expr):
    """z_n"""
    argument_types = ["int"]
    return_type = "int"
    
    def __init__(self, i):
        self.i = i

    def __str__(self):
        return f"Zn('{self.i}')"

    def pretty_print(self):
        return f"z_n[{self.i}]"

    def eval(self, env):
        i = self.i.eval(env)
        # print(f"eval {self.pretty_print()}, i={i}, len={len(env['z_n'])}")
        assert isinstance(i, int)
        return env["z_n"][i]

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

class If(Expr):
    argument_types = ["bool","Bitmap","Bitmap"]
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
        assert isinstance(test, bool) and isinstance(yes, int) and isinstance(no, int)
        return yes if test else no

class Point(Expr):
    argument_types = ["int", "int"]
    return_type = "Point"

    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __str__(self):
        return f"Point({self.x}, {self.y})"

    def pretty_print(self):
        return f"({self.x.pretty_print()}, {self.y.pretty_print()})"

    def eval(self, env):
        x = self.x.eval(env)
        y = self.y.eval(env)
        assert isinstance(x, int) and isinstance(y, int) and \
            0 <= x <= BMP_WIDTH and 0 <= y <= BMP_HEIGHT
        return Point(x, y)

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

    argument_types = ["Point", "Point"]
    return_type = "Bitmap"
    
    def __init__(self, p1, p2):
        # assert isinstance(p1, Point) and isinstance(p2, Point)
        self.p1 = p1
        self.p2 = p2
        
    def __str__(self):
        return f"Rect({self.p1}, {self.p2})"

    def pretty_print(self):
        return f"({self.p1.pretty_print()}, {self.p2.pretty_print()})"

    def eval(self, env):
        p1 = self.p1.eval(env)
        p2 = self.p2.eval(env)
        # check invariants
        assert 0 <= p1.x < p2.x <= BMP_WIDTH and 0 <= p1.y < p2.y <= BMP_HEIGHT
        return Bitmap([[p1.x <= x < p2.x and p1.y <= y < p2.y
                        for x in range(BMP_WIDTH)]
                       for y in range(BMP_HEIGHT)])

class Program(Expr):
    argument_types = ["Bitmap", "Bitmap"]
    return_type = "Bitmap"
    
    def __init__(self, left, right):
        self.left = left
        self.right = right
        
    def __str__(self):
        return f"Program({self.left}, {self.right})"

    def pretty_print(self):
        return (f"{self.left.pretty_print()}; {self.right.pretty_print()}")

    def eval(self, env):
        left = self.left.eval(env)
        right = self.right.eval(env)
        return left.OR(right)

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
        (Rect(Point(Num(0),
                    Num(0)), 
              Point(Num(1),
                    Num(2))),
         lambda zb, zn: Bitmap.from_img(["#___",
                                         "#___",
                                         "____",
                                         "____",])),
        (Program(Rect(Point(Num(0),
                            Num(0)), 
                      Point(Num(1),
                            Num(1))),
                 Rect(Point(Num(2),
                            Num(3)), 
                      Point(Num(4),
                            Num(4)))),
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
    expr = Rect(Point(Num(0),Num(0)), 
                Point(Num(1),Num(1)))
    out = expr.eval({})
    expected = Bitmap.from_img(["#___",
                                "____",
                                "____",
                                "____"])
    assert expected == out, f"test_render failed:\n expected={expected},\n out={out}"

    # (1,0), (3,3)
    expr = Rect(Point(Zn(Num(0)), Num(0)), 
                Point(Plus(Num(2), Num(1)), Num(3))) 
    out = expr.eval({'z_n': [1,2,3]})
    expected = Bitmap.from_img(["_##_",
                                "_##_",
                                "_##_",
                                "____"])
    assert expected == out, f"test_render failed:\n expected={expected},\n out={out}"
    print(" [+] passed test_render")

if __name__ == '__main__':
    test_eval()
    test_render()
    
