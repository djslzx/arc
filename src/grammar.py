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
        assert isinstance(i, int)
        self.i = i

    def __str__(self):
        return f"Zb('{self.i}')"

    def pretty_print(self):
        return f"z_b[{self.i}]"

    def eval(self, env):
        return env["z_b"][self.i]

class Number(Expr):
    return_type = "int"
    argument_types = []
    
    def __init__(self, n):
        self.n = n

    def __str__(self):
        return f"Number({self.n})"

    def pretty_print(self):
        return str(self.n)

    def eval(self, env):
        return self.n

class Zn(Expr):
    """z_n"""
    return_type = "int"
    argument_types = ["int"]
    
    def __init__(self, i):
        assert isinstance(i, int)
        self.i = i

    def __str__(self):
        return f"Zn('{self.i}')"

    def pretty_print(self):
        return f"z_n[{self.i}]"

    def eval(self, env):
        return env["z_n"][self.i]

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

class Div(Expr):
    return_type = "int"
    argument_types = ["int","int"]
    
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Div({self.x}, {self.y})"

    def pretty_print(self):
        return f"(/ {self.x.pretty_print()} {self.y.pretty_print()})"

    def eval(self, env):
        x = self.x.eval(env)
        y = self.y.eval(env)
        assert isinstance(x, int) and isinstance(y, int)
        return x // y

class LessThan(Expr):
    return_type = "bool"
    argument_types = ["int","int"]
    
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"LessThan({self.x}, {self.y})"

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
    return_type = "tuple(int)"
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
        return (x, y)

class Rect(Expr):
    return_type = "tuple(tuple(int), tuple(int))"
    argument_types = ["tuple(int)","tuple(int)"]
    
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        
    def __str__(self):
        return f"Rect({self.p1}, {self.p2})"

    def pretty_print(self):
        return f"({self.p1.pretty_print()}, {self.p2.pretty_print()})"

    def eval(self, env):
        p1 = self.p1.eval(env)
        p2 = self.p2.eval(env)
        assert isinstance(p1, int) and isinstance(p2, int)
        return (p1, p2)
