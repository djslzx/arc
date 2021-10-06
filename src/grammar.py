import util

# input dimensions
IMG_WIDTH=8
IMG_HEIGHT=8

# constants for z_n, z_b
Z_LO = -8                       # min poss value in z_n
Z_HI = 8                        # max poss value in z_n
Z_SIZE = 16                     # length of z_n, z_b 

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
    return_type = "bool"
    argument_types = []
    
    def __init__(self): pass

    def __str__(self):
        return "False"

    def pretty_print(self):
        return "False"

    def eval(self, env):
        return False
    
class Zb(Expr):
    return_type = "bool"
    argument_types = ["int"]
    
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
    return_type = "int"
    argument_types = []
    
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
    return_type = "int"
    argument_types = ["int"]
    
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
    return_type = "int"
    argument_types = ["int","int"]
    
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
    return_type = "int"
    argument_types = ["int","int"]
    
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
    return_type = "int"
    argument_types = ["int","int"]
    
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
    return_type = "bool"
    argument_types = ["int","int"]
    
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
    return_type = "bool"
    argument_types = ["bool","bool"]
    
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
    return_type = "bool"
    argument_types = ["bool"]
    
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
    return_type = "int"
    argument_types = ["bool","int","int"]
    
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
    return_type = "Point"
    argument_types = ["int", "int"]

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
        assert isinstance(x, int) and isinstance(y, int)
        return Point(x, y)

class Rect(Expr):
    return_type = "Rect"
    argument_types = ["Point", "Point"]
    
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
        # assert isinstance(p1, tuple(int)) and isinstance(p2, tuple(int))
        return Rect(p1, p2)

    def contains(self, env, p):
        """
        Check whether `self` contains the point `p`
        """
        x, y = p.x, p.y
        p1 = self.p1.eval(env)
        p2 = self.p2.eval(env)
        return p1.x <= x <= p2.x and p1.y <= y <= p2.y

    def render(self, env):
        """
        Render an expression as a 'pixel image' (a boolean matrix)
        """
        return [[self.contains(env, Point(x,y))
                 for x in range(IMG_WIDTH)]
                for y in range(IMG_HEIGHT)]

class Program(Expr):
    return_type = "list"        # boolean matrix
    argument_types = []
    
    def __init__(self, rects):
        assert all(isinstance(r, Rect) for r in rects)
        self.rects = rects
        
    def __str__(self):
        return f"Program({[r for r in self.rects]})"

    def pretty_print(self):
        return ";\n".join([r.pretty_print() 
                           for r in self.rects])

    def eval(self, env):
        # TODO: enforce invariant that rects don't overlap
        return [[any(r.contains(env, Point(x,y)) for r in self.rects)
                 for x in range(IMG_WIDTH)]
                for y in range(IMG_HEIGHT)]

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
        (Program([Rect(Point(Num(0),Num(0)), Point(Num(1),Num(1))),
                  Rect(Point(Num(3),Num(3)), Point(Num(4),Num(6)))]),
         lambda zb, zn: util.img_to_bool_matrix(
             ["##______",
              "##______",
              "________",
              "___##___",
              "___##___",
              "___##___",
              "___##___",
              "________",]
         ))
    ]
    for expr, correct_semantics in tests:
        for x in range(10):
            for y in range(10):
                out = expr.eval({"z_b":[], "z_n":[x,y]})
                expected = correct_semantics([], [x,y])
                assert out == expected, f"failed eval test:\n"\
                    f" expected={util.bool_matrix_to_img(expected)}\n"\
                    f" out={util.bool_matrix_to_img(out)}"
    print(" [+] eval passes checks")

def test_render():
    # (0,0), (1,1)
    expr = Rect(Point(Num(0),Num(0)), 
                Point(Num(1),Num(1)))
    out = expr.render({})
    expected = util.img_to_bool_matrix(
        ["##______",
         "##______",
         "________",
         "________",
         "________",
         "________",
         "________",
         "________"])
    assert expected == out, f"test_render failed:\n expected={expected},\n out={out}"

    # (1,0), (3,3)
    expr = Rect(Point(Zn(Num(0)), 
                      Num(0)), 
                Point(Plus(Num(2), Num(1)), 
                      Num(3))) 
    out = expr.render({'z_n': [1,2,3]})
    expected = util.img_to_bool_matrix( 
        ["_###____",
         "_###____",
         "_###____",
         "_###____",
         "________",
         "________",
         "________",
         "________"])
    assert expected == out, f"test_render failed:\n expected={expected},\n out={out}"
    print(" [+] passed test_render")

if __name__ == '__main__':
    test_eval()
    test_render()
    
