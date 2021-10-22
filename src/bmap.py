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

    def as_pts(self):
        return [(x,y) 
                for x in range(self.width)
                for y in range(self.height)
                if self.mat[y][x]]

    def num_on(self):
        """Number of pixels toggled 'on'"""
        return sum(self.mat[y][x]
                   for x in range(self.width)
                   for y in range(self.height))

    def pretty_print(self):
        return "\n".join(bool_matrix_to_img(self.mat))

    def apply(self, op, other):
        assert isinstance(self, Bitmap) and isinstance(other, Bitmap)
        assert self.height == other.height and self.width == other.width
        return Bitmap(
            [[op(self.mat[y][x], other.mat[y][x])
              for x in range(self.width)]
             for y in range(self.height)])

    def intersect(self, other):
        return self.apply(lambda x,y: x and y, other)

    def union(self, other):
        return self.apply(lambda x,y: x or y, other)
    
    def px_dist(self, other):
        assert self.width == other.width and self.height == other.height
        return Bitmap.intersect(self, other).num_on()

    def iou_dist(self, other):
        assert self.width == other.width and self.height == other.height
        return Bitmap.intersect(self, other).num_on()/Bitmap.union(self, other).num_on()

    def dist(self, other):
        return self.iou_dist(other)

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
        actual = Bitmap(x).OR(Bitmap(y))
        assert actual == Bitmap(expected), \
            f"test failed: {x} OR {y} = {actual}, expected={expected}"

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
        actual = Bitmap(x).AND(Bitmap(y))
        assert actual == Bitmap(expected), \
            f"test failed: {x} AND {y} = {actual}, expected={expected}"

if __name__ == '__main__':
    test_img_to_bool_matrix()
    test_bool_matrix_to_img()
    test_bitmap_or()
    test_bitmap_and()
