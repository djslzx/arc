class Expression():

    def evaluate(self, environment):
        assert False, f"not implemented for {self}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self): return hash(str(self))

    def __ne__(self, other): return str(self) != str(other)

    def __gt__(self, other): return str(self) > str(other)

    def __lt__(self, other): return str(self) < str(other)

class FALSE(Expression):
    return_type = "bool"
    argument_types = []
    
    def __init__(self): pass

    def __str__(self):
        return "False"

    def pretty_print(self):
        return "False"

    def evaluate(self, environment):
        return False
    
class Number(Expression):
    return_type = "int"
    argument_types = []
    
    def __init__(self, n):
        self.n = n

    def __str__(self):
        return f"Number({self.n})"

    def pretty_print(self):
        return str(self.n)

    def evaluate(self, environment):
        return self.n

class NumberVariable(Expression):
    return_type = "int"
    argument_types = []
    
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"NumberVariable('{self.name}')"

    def pretty_print(self):
        return self.name

    def evaluate(self, environment):
        return environment[self.name]

class Plus(Expression):
    return_type = "int"
    argument_types = ["int","int"]
    
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Plus({self.x}, {self.y})"

    def pretty_print(self):
        return f"(+ {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        assert isinstance(x, int) and isinstance(y, int)
        return x + y

class Minus(Expression):
    return_type = "int"
    argument_types = ["int","int"]
    
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Minus({self.x}, {self.y})"

    def pretty_print(self):
        return f"(- {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        assert isinstance(x, int) and isinstance(y, int)
        return x - y

class Times(Expression):
    return_type = "int"
    argument_types = ["int","int"]
    
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Times({self.x}, {self.y})"

    def pretty_print(self):
        return f"(* {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        assert isinstance(x, int) and isinstance(y, int)
        return x * y

class Div(Expression):
    return_type = "int"
    argument_types = ["int","int"]
    
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Div({self.x}, {self.y})"

    def pretty_print(self):
        return f"(/ {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        assert isinstance(x, int) and isinstance(y, int)
        return x // y

class LessThan(Expression):
    return_type = "bool"
    argument_types = ["int","int"]
    
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"LessThan({self.x}, {self.y})"

    def pretty_print(self):
        return f"(< {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        assert isinstance(x, int) and isinstance(y, int)
        return x < y

class And(Expression):
    return_type = "bool"
    argument_types = ["bool","bool"]
    
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"And({self.x}, {self.y})"

    def pretty_print(self):
        return f"(and {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        assert isinstance(x, bool) and isinstance(y, bool)
        return x and y

class Not(Expression):
    return_type = "bool"
    argument_types = ["bool"]
    
    def __init__(self, x):
        self.x = x

    def __str__(self):
        return f"Not({self.x})"

    def pretty_print(self):
        return f"(not {self.x.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        assert isinstance(x, bool)
        return not x

class If(Expression):
    return_type = "int"
    argument_types = ["bool","int","int"]
    
    def __init__(self, test, yes, no):
        self.test, self.yes, self.no = test, yes, no

    def __str__(self):
        return f"If({self.test}, {self.yes}, {self.no})"

    def pretty_print(self):
        return f"(if {self.test.pretty_print()} {self.yes.pretty_print()} {self.no.pretty_print()})"

    def evaluate(self, environment):
        test = self.test.evaluate(environment)
        yes = self.yes.evaluate(environment)
        no = self.no.evaluate(environment)
        assert isinstance(test, bool) and isinstance(yes, int) and isinstance(no, int)
        return yes if test else no
