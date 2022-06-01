"""
Generate a plot of the widths and heights of ARC problems.
"""

import json
from glob import glob
from typing import List, Dict, Tuple
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import torch as T
import pdb

import viz


Mat = List[List]
TaskPair = Dict[str, Mat]
Task = Dict[str, List[TaskPair]]

def read_arc_file(filename: str) -> Tuple[str, Task]:
    with open(filename, 'r') as f:
        return filename, json.load(f)

def read_arc_files(filenames: List[str]) -> List[Tuple[str, Task]]:
    return [read_arc_file(filename) for filename in filenames]

def arc_task_as_tensors(task: Task) -> List[T.Tensor]:
    """
    Reads in an ARC task ('train'/'test' -> ['input'/'output' -> List[List]])
    and returns a set of bitmaps.
    """
    bitmaps = []
    for instance in task['train'] + task['test']:
        for k in ['input', 'output']:
            mat = instance[k]
            bitmap = T.tensor(mat)
            bitmaps.append(bitmap)
    return bitmaps

def arc_tasks_as_tensors(tasks: List[Task]) -> List[T.Tensor]:
    return [t 
            for task in tasks 
            for t in arc_task_as_tensors(task)]

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

def dims(arc_tasks: List[Task], pair_keys: List[str]) -> List[Dict[str, int]]:
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
    ARC_DIR = '../ARC/data/*'
    TASK_NAMES = [
        "025d127b", "150deff5", "1caeab9d", "1e0a9b12", "1f876c06",
        "1fad071e", "2281f1f4", "228f6490", "31aa019c", "39a8645d",
        "3aa6fb7a", "3af2c5a8", "444801d8", "47c1f68c", "5c0a986e",
        "6150a2bd", "62c24649", "6773b310", "6c434453", "6e82a1ae",
        "7c008303", "7ddcd7ec", "97999447", "99b1bc43", "a3325580",
        "a5313dff", "a78176bb", "a87f7484", "ac0a08a4", "b8cdaf2b",
        "b94a9452", "c8f0f002", "d631b094", "d9fac9be", "ded97339",
        "e179c5f4", "e5062a87", "e50d258f", "f76d97a5", "f9012d9b",
    ]
    filenames = [name 
                 for task_name in TASK_NAMES
                 for name in glob(f'{ARC_DIR}/{task_name}.json')]

    tasks = read_arc_files(filenames)
    for name, task in tasks:
        bitmaps = arc_task_as_tensors(task)
        for b in bitmaps:
            viz.viz(b, title=name)

    # dimensions = dims(tasks, ['input', 'output'])
    # print(plot_dims(dimensions))
    # n_dims = len(dimensions)
    # for n in range(18, 32):
    #     n_filtered = n_tasks_within_dim(n, n, dimensions)
    #     print(n, n_filtered)
