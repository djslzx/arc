"""
Generate a plot of the widths and heights of ARC problems.
"""

import json
from glob import glob
from typing import List, Dict
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import pdb

def read_arc_file(filename: str) -> Dict:
    with open(filename, 'r') as f:
        return json.load(f)

def read_arc_files(filename_glob: str) -> List[Dict]:
    filenames = glob(filename_glob)
    return [read_arc_file(filename) for filename in filenames]

def dim(arc_task: Dict, pair_keys: List[str]) -> List[Dict[str, int]]:
    """
    Measure the dimensions (height, width) of the bitmap instances in the task.
    """
    return [
        {
            'height': len(mat),
            'width': len(mat[0])
        }
        for pair in arc_task['train'] + arc_task['test']
        for mat in [pair[k] for k in pair_keys]
    ]

def dims(arc_tasks: List[Dict], pair_keys: List[str]) -> List[Dict[str, int]]:
    return list(it.chain.from_iterable(dim(task, pair_keys) for task in arc_tasks))

def plot_dims(dims: List[Dict[str, int]]):
    heights = [x['height'] for x in dims]
    widths = [x['width'] for x in dims]

    mat = np.zeros((max(heights) + 1, max(widths) + 1))
    for h, w in zip(heights, widths):
        mat[h, w] += 1

    plt.imshow(mat, cmap='gray')
    plt.scatter(widths, heights, s=2, alpha=0.5, color='white')
    plt.title('all')
    plt.show()

def n_tasks_within_dim(height: int, width: int, dims: List[Dict]) -> int:
    """
    Count the number of bitmaps whose dimensions are within height, width.
    """
    n = 0
    for dim in dims:
        n += dim['height'] <= height and dim['width'] <= width
    return n

if __name__ == '__main__':
    ARC_GLOB = '../chollet/data/*/*.json'
    tasks = read_arc_files(ARC_GLOB)
    dimensions = dims(tasks, ['input', 'output'])
    print(plot_dims(dimensions))
    n_dims = len(dimensions)
    for n in range(18, 32):
        n_filtered = n_tasks_within_dim(n, n, dimensions)
        print(n, n_filtered)
