"""
A simple probabilistic string grammar.

AB0 expands to ABA, ABB, ... depending on the choice of random vector.

"""

import torch as T
from typing import List

def evaluate(s: str, Z: List[List[str]]) -> List[str]:
    def eval_once(s: str, z: List[str]) -> str:
        out = ''
        for c in s:
            try:
                i = int(c)
                out += z[i]
            except ValueError:
                out += c
        return out

    return [eval_once(s, z) for z in Z]

def test_evaluate():
    cases = [
        ('ab', [['d', 'e', 'f'], ['x', 'y', 'z']],
         ['ab', 'ab']),
        ('ab0', [['d', 'e', 'f'], ['x', 'y', 'z']],
         ['abd', 'abx']),
        ('a0b1c2', [['d', 'e', 'f'], ['x', 'y', 'z']],
         ['adbecf', 'axbycz']),
    ]
    for *inputs, target in cases:
        output = evaluate(*inputs)
        assert output == target, f'input={inputs}, expected={target}, actual={output}'
    print ('[+] test_evaluate passed')
    

if __name__ == '__main__':
    test_evaluate()