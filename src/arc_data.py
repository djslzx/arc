"""
Generate a plot of the widths and heights of ARC problems.
"""

import json
from glob import glob
from typing import List, Dict, Tuple, Set
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import torch as T
import pdb

import grammar
import viz
import util

Mat = List[List]
TaskPair = Dict[str, Mat]
Task = Dict[str, List[TaskPair]]

ARC_DATA_DIR = '/home/djsl/Research/arc/ARC/data/*'
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
SELECTED_TASK_NAMES = [
    '025d127b', '150deff5', '1caeab9d', '1e0a9b12', '39a8645d',
    '3aa6fb7a', '3af2c5a8', '444801d8', '5c0a986e', '6150a2bd',
    '6c434453', '6e82a1ae', '7c008303', '7ddcd7ec', '99b1bc43',
    'a3325580',
]
SEQ_FEASIBLE_TASK_NAMES = [
    '025d127b', '150deff5', '1caeab9d', '3af2c5a8', '47c1f68c', '5c0a986e',
    '6150a2bd', '62c24649', '6c434453', '7c008303', '7ddcd7ec', '99b1bc43',
    'b8cdaf2b', 'b94a9452', 'c8f0f002', 'd631b094', 'e179c5f4', 'f76d97a5',
]

def read_arc_file(filename: str) -> Task:
    with open(filename, 'r') as f:
        return json.load(f)

def read_arc_files(filenames: List[str]) -> List[Task]:
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

def map_task(arc_task: Task, f, domain: List[str]) -> List:
    return [f(mat)
            for pair in arc_task['train'] + arc_task['test']
            for mat in [pair[k] for k in domain]]

def colors(arc_tasks: List[Task], domain: List[str]) -> Set[int]:
    """
    Record the colors used in the list of ARC tasks.
    """
    def flatten(mat): return [cell for row in mat for cell in row]
    seen = set()
    for arc_task in arc_tasks:
        for mat in map_task(arc_task, flatten, domain):
            seen |= set(mat)
    return seen

def dim(arc_task: Task, domain: List[str]) -> List[Dict[str, int]]:
    """
    Measure the dimensions (height, width) of the bitmap instances in the task.
    """
    return map_task(
        arc_task,
        lambda mat: {
            'height': len(mat),
            'width': len(mat[0])
        },
        domain
    )

def dims(arc_tasks: List[Task], domain: List[str]) -> List[Dict[str, int]]:
    return list(it.chain.from_iterable(dim(task, domain) for task in arc_tasks))

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

def get_fnames(names: List[str]) -> List[str]:
    return [name
            for task_name in names
            for name in glob(f'{ARC_DATA_DIR}/{task_name}.json')]

def get_dims(names: List[str], domain: List[str]):
    filenames = get_fnames(names)
    tasks = read_arc_files(filenames)
    return dims(tasks, domain)

def plot_arc_dimensions(names: List[str]):
    dimensions = get_dims(names, domain=['input', 'output'])
    plot_dims(dimensions)

def pad_bitmaps(bitmaps: List[T.Tensor]) -> List[T.Tensor]:
    return [util.pad_mat(bitmap, h=grammar.B_H, w=grammar.B_W)
            for bitmap in bitmaps]

def task_bitmaps(names: List[str]) -> List[T.Tensor]:
    filenames = get_fnames(names)
    bitmaps = []
    for task in read_arc_files(filenames):
        bitmaps.extend(pad_bitmaps(arc_task_as_tensors(task)))
    return bitmaps

def named_task_bitmaps(task_names: List[str]) -> Tuple[List[T.Tensor], List[str]]:
    bitmaps = []
    bitmap_names = []
    for task_name, task in zip(task_names, read_arc_files(get_fnames(task_names))):
        task_tensors = arc_task_as_tensors(task)
        bitmaps.extend(pad_bitmaps(task_tensors))
        bitmap_names.extend([f'{task_name}-{i}' for i in range(len(task_tensors))])
    return bitmaps, bitmap_names

def survey_colors():
    filenames = [name for task_name in TASK_NAMES
                 for name in glob(f'{ARC_DATA_DIR}/{task_name}.json')]
    tasks = [task for name, task in read_arc_files(filenames)]
    print(colors(tasks, ['input', 'output']))


if __name__ == '__main__':
    # print(selected_task_bitmaps())
    # survey_colors()
    plot_arc_dimensions(names=SEQ_FEASIBLE_TASK_NAMES)
    dims = get_dims(names=SEQ_FEASIBLE_TASK_NAMES, domain=['input', 'output'])
    print(n_tasks_within_dim(10, 10, dims))
    print(n_tasks_within_dim(14, 14, dims))
