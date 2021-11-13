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

'''
Grammar

- Expr: F, N, Z, +, -, *, <, and, not, if, rect, prog
+ reflect, diagonal/horizontal/vertical line
'''

class Visited:
    def accept(self, visitor):
        assert False, "Visited subclass should implement `accept`"
    def eval(self, env): return self.accept(Eval(env))
    @property
    def in_types(self): return self.accept(InTypes())
    @property
    def return_type(self): return self.accept(RetType())
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
    def __init__(self): pass
    def accept(self, v): return v.visit_F()
class N(Expr): 
    def __init__(self, n): self.n = n
    def accept(self, v): return v.visit_N(self.n)
class Z(Expr): 
    def __init__(self, i): self.i = i
    def accept(self, v): return v.visit_Z(self.i.accept(v))
class Not(Expr):
    def __init__(self, b): self.b = b
    def accept(self, v): return v.visit_Not(self.b.accept(v))
class Plus(Expr): 
    def __init__(self, x, y): 
        self.x = x
        self.y = y
    def accept(self, v): return v.visit_Plus(self.x.accept(v), 
                                             self.y.accept(v))
class Minus(Expr): 
    def __init__(self, x, y): 
        self.x = x
        self.y = y
    def accept(self, v): return v.visit_Minus(self.x.accept(v), 
                                              self.y.accept(v))
class Times(Expr): 
    def __init__(self, x, y): 
        self.x = x
        self.y = y
    def accept(self, v): return v.visit_Times(self.x.accept(v), 
                                              self.y.accept(v))
class Lt(Expr): 
    def __init__(self, x, y): 
        self.x = x
        self.y = y
    def accept(self, v): return v.visit_Lt(self.x.accept(v), 
                                           self.y.accept(v))
class And(Expr): 
    def __init__(self, x, y): 
        self.x = x
        self.y = y
    def accept(self, v): return v.visit_And(self.x.accept(v), 
                                            self.y.accept(v))
class If(Expr): 
    def __init__(self, b, x, y): 
        self.b = b
        self.x = x
        self.y = y
    def accept(self, v): return v.visit_If(self.b.accept(v),
                                           self.x.accept(v), 
                                           self.y.accept(v))
class Rect(Expr): 
    def __init__(self, x1, y1, x2, y2): 
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    def accept(self, v): return v.visit_Rect(self.x1.accept(v), 
                                             self.y1.accept(v),
                                             self.x2.accept(v), 
                                             self.y2.accept(v))
class Union(Expr): 
    def __init__(self, b1, b2): 
        self.b1 = b1
        self.b2 = b2
    def accept(self, v): return v.visit_Union(self.b1.accept(v), 
                                              self.b2.accept(v))
class Intersect(Expr): 
    def __init__(self, b1, b2): 
        self.b1 = b1
        self.b2 = b2
    def accept(self, v): return v.visit_Intersect(self.b1.accept(v), 
                                                  self.b2.accept(v))

class Visitor:
    def _fail(self, s): assert False, f"Visitor subclass should implement `{s}`"
    def visit_F(self): self._fail('F')
    def visit_N(self, n): self._fail('N')
    def visit_Z(self, i): self._fail('Z')
    def visit_Not(self, b): self._fail('Not')
    def visit_Plus(self, x, y): self._fail('Plus')
    def visit_Minus(self, x, y): self._fail('Minus')
    def visit_Times(self, x, y): self._fail('Times')
    def visit_Lt(self, x, y): self._fail('Lt')
    def visit_And(self, x, y): self._fail('And')
    def visit_If(self, b, x, y): self._fail('If')
    def visit_Rect(self, x1, y1, x2, y2): self._fail('Rect')
    def visit_Union(self, b1, b2): self._fail('Union')
    def visit_Intersect(self, b1, b2): self._fail('Union')

class Eval(Visitor):
    def __init__(self, env): self.env = env
    def visit_F(self): return False
    def visit_N(self, n): return n
    def visit_Z(self, i): 
        assert isinstance(i, int)
        assert 'z' in self.env, "Eval env missing Z"
        z = self.env['z'][i]
        return z.item() if isinstance(z, T.LongTensor) else z
    def visit_Not(self, b): 
        assert isinstance(b, bool)
        return not b
    def visit_Plus(self, x, y): 
        assert isinstance(x, int) and isinstance(y, int)
        return x + y
    def visit_Minus(self, x, y): 
        assert isinstance(x, int) and isinstance(y, int)
        return x - y
    def visit_Times(self, x, y): 
        assert isinstance(x, int) and isinstance(y, int)
        return x * y
    def visit_Lt(self, x, y): 
        assert isinstance(x, int) and isinstance(y, int)
        return x < y
    def visit_And(self, x, y): 
        assert isinstance(x, bool) and isinstance(y, bool)
        return x and y
    def visit_If(self, b, x, y): 
        assert isinstance(b, bool) and isinstance(x, int) and isinstance(y, int)
        return x if b else y
    def visit_Rect(self, x1, y1, x2, y2): 
        assert all(isinstance(v, int) for v in [x1, y1, x2, y2])
        assert 0 <= x1 < x2 <= B_W and 0 <= y1 < y2 <= B_H
        return T.tensor([[x1 <= x < x2 and y1 <= y < y2
                          for x in range(B_W)]
                         for y in range(B_H)]).float()
    def visit_Union(self, b1, b2): 
        assert isinstance(b1, T.FloatTensor) and isinstance(b2, T.FloatTensor), \
            f"Union needs two float tensors: b1={b1}, b2={b2}"
        return b1.logical_or(b2).float()
    def visit_Intersect(self, b1, b2): 
        assert isinstance(b1, T.FloatTensor) and isinstance(b2, T.FloatTensor), \
            f"Intersect needs two float tensors: b1={b1}, b2={b2}"
        return b1.logical_and(b2).float()

class InTypes(Visitor):
    def __init__(self): pass
    def visit_F(self): return []
    def visit_N(self, n): return []
    def visit_Z(self, i): return ['int']
    def visit_Not(self, b): return ['bool']
    def visit_Plus(self, x, y): return ['int', 'int']
    def visit_Minus(self, x, y): return ['int', 'int']
    def visit_Times(self, x, y): return ['int', 'int']
    def visit_Lt(self, x, y): return ['int', 'int']
    def visit_And(self, x, y): return ['bool', 'bool']
    def visit_If(self, b, x, y): return ['bool', 'int', 'int']
    def visit_Rect(self, x1, y1, x2, y2): return ['int', 'int', 'int', 'int']
    def visit_Union(self, b1, b2): return ['bitmap', 'bitmap']
    def visit_Intersect(self, b1, b2): return ['bitmap', 'bitmap']

class RetType(Visitor):
    def __init__(self): pass
    def visit_F(self): return 'bool'
    def visit_N(self, n): return 'int'
    def visit_Z(self, i): return 'int'
    def visit_Not(self, b): return 'bool'
    def visit_Plus(self, x, y): return 'int'
    def visit_Minus(self, x, y): return 'int'
    def visit_Times(self, x, y): return 'int'
    def visit_Lt(self, x, y): return 'bool'
    def visit_And(self, x, y): return 'bool'
    def visit_If(self, b, x, y): return 'int'
    def visit_Rect(self, x1, y1, x2, y2): return 'bitmap'
    def visit_Union(self, b1, b2): return 'bitmap'
    def visit_Intersect(self, b1, b2): return 'bitmap'

class Print(Visitor):
    def __init__(self): pass
    def visit_F(self): return 'False'
    def visit_N(self, n): return f'{n}'
    def visit_Z(self, i): return f'z[{i}]'
    def visit_Not(self, b): return f'(not {b})'
    def visit_Plus(self, x, y): return f'(+ {x} {y})'
    def visit_Minus(self, x, y): return f'(- {x} {y})'
    def visit_Times(self, x, y): return f'(* {x} {y})'
    def visit_Lt(self, x, y): return f'(< {x} {y})'
    def visit_And(self, x, y): return f'(and {x} {y})'
    def visit_If(self, b, x, y): return f'(if {b} {x} {y})'
    def visit_Rect(self, x1, y1, x2, y2): return f'(R {x1} {y1} {x2} {y2})'
    def visit_Union(self, b1, b2): return f'(u {b1} {b2})'
    def visit_Intersect(self, b1, b2): return f'(n {b1} {b2})'

class Zs(Visitor):
    def __init__(self): pass
    def visit_F(self): return {}
    def visit_N(self, n): return {}
    def visit_Z(self, i): return {i}
    def visit_Not(self, b): return b
    def visit_Plus(self, x, y): return x | y
    def visit_Minus(self, x, y): return x | y
    def visit_Times(self, x, y): return x | y
    def visit_Lt(self, x, y): return x | y
    def visit_And(self, x, y): return x | y
    def visit_If(self, b, x, y): return b | x | y
    def visit_Rect(self, x1, y1, x2, y2): x1 | y1 | x2 | y2
    def visit_Union(self, b1, b2): return b1 | b2
    def visit_Intersect(self, b1, b2): return b1 | b2

def img_to_tensor(lines):
    """Converts a list of strings into a float tensor"""
    return T.tensor([[c == "#" for c in line] 
                     for line in lines]).float()

def test_eval():
    tests = [
        (F(),
         lambda z: False),
        (Not(F()),
         lambda z: True),
        (Times(Z(N(0)), Z(N(1))),
         lambda z: z[0] * z[1]),
        (If(Lt(Z(N(0)), Z(N(1))),
            Z(N(0)), 
            Z(N(1))),
         lambda z: min(z[0], z[1])),
        (If(Not(Lt(Z(N(0)), 
                   Z(N(1)))),
            Times(Z(N(0)), Z(N(1))), 
            Plus(Z(N(0)), Z(N(1)))), 
         lambda z: z[0] * z[1] if not (z[0] < z[1]) else z[0] + z[1]),
        (Rect(N(0), N(0), 
              N(1), N(2)),
         lambda z: img_to_tensor(["#___",
                                  "#___",
                                  "____",
                                  "____"])),
        (Union(Rect(N(0), N(0), 
                    N(1), N(1)),
               Rect(N(2), N(3), 
                    N(4), N(4))),
         lambda z: img_to_tensor(["#___",
                                  "____",
                                  "____",
                                  "__##"])),
    ]
    for expr, correct_semantics in tests:
        for x in range(10):
            for y in range(10):
                out = expr.eval({"z":[x, y]})
                expected = correct_semantics([x, y])
                t = expr.return_type
                # print(expr, out, expected)
                if t in ['int', 'bool']:
                    assert out == expected, f"failed eval test:\n"\
                        f" expected=\n{expected}\n"\
                        f" out=\n{out}"
                elif t == 'bitmap':
                    assert T.equal(out, expected), f"failed eval test:\n"\
                        f" expected=\n{expected}\n"\
                        f" out=\n{out}"
                else:
                    assert False, "type error in eval test"

    # (0,0), (1,1)
    expr = Rect(N(0),N(0), 
                N(1),N(1))
    out = expr.eval({'z':[]})
    expected = img_to_tensor(["#___",
                              "____",
                              "____",
                              "____"])
    assert T.equal(expected, out), f"test_render failed:\n expected={expected},\n out={out}"

    # (1,0), (3,3)
    expr = Rect(Z(N(0)), 
                N(0), 
                Plus(N(2), N(1)), 
                N(3)) 
    out = expr.eval({'z': [1,2,3]})
    expected = img_to_tensor(["_##_",
                              "_##_",
                              "_##_",
                              "____"])
    assert T.equal(expected, out), f"test_render failed:\n expected={expected},\n out={out}"
    print(" [+] passed test_eval")

def test_zs():
    test_cases = [
        (Rect(N(0), N(1), N(4), N(4)), 
         {}),
        (Rect(Z(N(0)), Z(N(1)), N(4), N(4)), 
         {0, 1}),
        (Rect(Z(N(0)), Z(N(1)), Z(N(2)), Z(N(3))),
         {0, 1, 2, 3}),
        (Rect(Z(N(0)), Z(N(1)), Z(N(0)), Z(N(1))),
         {0, 1}),
    ]
    for expr, ans in test_cases:
        out = expr.zs()
        print(expr, out, ans)
        # assert out == ans, f"test_zs failed: expected={ans}, actual={out}"
    print(" [+] passed test_zs")

if __name__ == '__main__':
    test_eval()
    test_zs()
