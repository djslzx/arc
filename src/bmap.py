import math
import torch as T

def img_to_bool_matrix(lines):
    """Converts an 'image' (a list of strings) into a row-major boolean matrix"""
    return [[c == "#" for c in line] 
            for line in lines]

def bool_matrix_to_img(mat):
    return ["".join(["#" if cell else "_" for cell in row])
            for row in mat]

class Bitmap:
    """
    A boolean 2D matrix
    """
    @staticmethod
    def from_img(s):
        return Bitmap(img_to_bool_matrix(s))

    @staticmethod
    def EMPTY(w, h):
        return Bitmap([[False] * w] * h)

    def __init__(self, mat):
        self.mat = mat
        self.height = len(mat)
        self.width = len(mat[0])
    
    def __str__(self):
        return f"Bitmap({self.as_pts()})"
    
    def __hash__(self):
        # return hash(str(self))
        return hash(tuple(tuple(row) for row in self.mat))

    def __eq__(self, other):
        return isinstance(other, Bitmap) and \
            self.height == other.height and \
            self.width == other.width and \
            self.mat == other.mat

    def empty(self):
        return not any(any(row) for row in self.mat)

    @property
    def dim(self):
        return (self.width, self.height)

    def as_pts(self):
        return [(x,y) 
                for x in range(self.width)
                for y in range(self.height)
                if self.mat[y][x]]

    def as_tensor(self):
        return T.tensor([[int(x) for x in row] for row in self.mat])

    def n_pts(self):
        """Number of pixels toggled 'on'"""
        return sum(sum(row) for row in self.mat)

    def pretty_print(self):
        return "\n".join(bool_matrix_to_img(self.mat))

    def apply(self, op, other):
        assert isinstance(self, Bitmap) and isinstance(other, Bitmap)
        assert self.height == other.height and self.width == other.width
        return Bitmap([
            [op(self.mat[y][x], other.mat[y][x])
             for x in range(self.width)]
            for y in range(self.height)])

    def intersect(self, other):
        return self.apply(lambda x,y: x and y, other)

    def union(self, other):
        return self.apply(lambda x,y: x or y, other)

    def xor(self, other):
        return self.apply(lambda x,y: x ^ y, other)
    
    def px_diff(self, other):
        return Bitmap.xor(self, other).n_pts()

    def iou_similarity(self, other):
        """Intersection over Union"""
        i = Bitmap.intersect(self, other).n_pts()
        u = Bitmap.union(self, other).n_pts()
        return i/u if u > 0 else 0

    def dist(self, other):
        if other is None: return math.inf
        return self.px_diff(other)

def test_img_to_bool_matrix():
    s = ["________",
         "________",
         "_____##_",
         "_____##_",
         "________",
         "________",
         "#####___",
         "#####___"]
    expected = [
        [False] * 8,
        [False] * 8,
        [False] * 5 + [True] * 2 + [False],
        [False] * 5 + [True] * 2 + [False],
        [False] * 8,
        [False] * 8,
        [True] * 5 + [False] * 3,
        [True] * 5 + [False] * 3,
    ]
    out = img_to_bool_matrix(s)
    assert out == expected, f"test_img_to_bool_matrix failed: expected={expected}, out={out}"
    print(" [+] passed test_img_to_bool_matrix")

def test_bool_matrix_to_img():
    tests = [
        ([[False]], ["_"]),
        ([[True]], ["#"]),
        ([[False, True],
          [False, True],
          [True, False],],
         ["_#",
          "_#",
          "#_"]),
    ]
    for x,y in tests:
        out = bool_matrix_to_img(x)
        assert out == y, f"test_bool_matrix_to_img failed: expected={y}, out={out}"

    b = [
        [False] * 8,
        [False] * 8,
        [False] * 5 + [True] * 2 + [False],
        [False] * 5 + [True] * 2 + [False],
        [False] * 8,
        [False] * 8,
        [True] * 5 + [False] * 3,
        [True] * 5 + [False] * 3,
    ]
    s = ["________",
         "________",
         "_____##_",
         "_____##_",
         "________",
         "________",
         "#####___",
         "#####___"]
    out = bool_matrix_to_img(b)
    assert out == s, f"test_bool_matrix_to_img failed: expected={s}, out={out}"
    print(" [+] passed test_bool_matrix_to_img")

def test_bitmap_or():
    tests = [
        ([[False]], [[False]], [[False]]),
        ([[True]], [[False]], [[True]]),
        ([[False]], [[True]], [[True]]),
        ([[True]], [[True]], [[True]]),
        ([[False, False],
          [True, True]], 
         [[True, False],
          [True, False]], 
         [[True, False],
          [True, True]]),
    ]
    for x, y, expected in tests:
        actual = Bitmap(x).union(Bitmap(y))
        assert actual == Bitmap(expected), \
            f"test failed: {x} union {y} = {actual}, expected={expected}"

def test_bitmap_and():
    tests = [
        ([[False]], [[False]], [[False]]),
        ([[True]], [[False]], [[False]]),
        ([[False]], [[True]], [[False]]),
        ([[True]], [[True]], [[True]]),
        ([[False, False],
          [True, True]], 
         [[True, False],
          [True, False]], 
         [[False, False],
          [True, False]]),
    ]
    for x, y, expected in tests:
        actual = Bitmap(x).intersect(Bitmap(y))
        assert actual == Bitmap(expected), \
            f"test failed: {x} intersect {y} = {actual}, expected={expected}"

def test_empty():
    tests = [
        ([[False]], True),
        ([[False, False], [False, False]], True),
        ([[False, False], [True, False]], False),
        ([[True, True], [True, True]], False),
    ]
    for x, expected in tests:
        actual = Bitmap(x).empty()
        assert actual == expected, \
            f"test failed: (empty {x})={actual}, expected={expected}"

if __name__ == '__main__':
    test_img_to_bool_matrix()
    test_bool_matrix_to_img()
    test_bitmap_or()
    test_bitmap_and()
    test_empty()
    print(" [+] passed all tests")
