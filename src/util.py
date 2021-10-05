# Utility fns

def img_to_bool_matrix(lines):
    """Converts an 'image' (a list of strings) into a row-major boolean matrix"""
    return [[c == "#" for c in line] 
            for line in lines]

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
    print("[+] passed test_img_to_bool_matrix")

if __name__ == '__main__':
    test_img_to_bool_matrix()
