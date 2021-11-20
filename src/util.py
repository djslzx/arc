import torch as T

def avg(it):
    s = 0
    n = 0
    for x in it:
        s += x
        n += 1
    return s/n

def sum_sq(it):
    return sum(x*x for x in it)

def chunks(l, k, n):
    size = len(l)
    for i in range(n):
        start = (i * k) % size
        end = (start + k) % size
        yield l[start:end] if start < end else l[start:] + l[:end]

def chunk_pairs(l, k, n):
    """
    Iterator over n k-elt chunks of list l, yielding pairs of adjacent chunks
    """
    size = len(l)
    for i in range(n):
        start = (i * k) % size
        mid = (start + k) % size
        end = (mid + k) % size
        
        yield (l[start:mid] if start < mid else l[start:] + l[:mid],
               l[mid:end] if mid < end else l[mid:] + l[:end])

def img_to_tensor(lines, w=-1, h=-1):
    """Converts a list of strings into a float tensor"""
    if h == -1: h = len(lines)
    if w == -1: w = len(lines[0])

    return T.tensor([[y < len(lines) and 
                      x < len(lines[0]) and 
                      lines[y][x] == '#' 
                      for x in range(w)] 
                     for y in range(h)]).float()

if __name__ == '__main__':
    print(img_to_tensor(['_#_',
                         '#_#',
                         '__#']))
    print(img_to_tensor(['_#_',
                         '#_#',
                         '__#'], 8, 8))
