# Utility fns

def img_to_bool_matrix(lines):
    """Converts an 'image' (a list of strings) into a row-major boolean matrix"""
    return [[c == "#" for c in line] 
            for line in lines]

def bool_matrix_to_img(mat):
    return ["".join(["#" if cell else "_" for cell in row])
            for row in mat]

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

if __name__ == '__main__':
    test_img_to_bool_matrix()
    test_bool_matrix_to_img()
