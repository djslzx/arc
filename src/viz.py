import sys
import json
import seaborn as sns
import torch as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from math import sqrt, floor

WIDTH=8
HEIGHT=10

sns.set_theme(style='white')

def clean(fname):
    name = '.'.join(fname.split('.')[:-1])
    name = name.split('/')[-1]
    return name

def label_cells(ax, grid):
    for (y,x), text in np.ndenumerate(grid):
        if text != 0:
            ax.text(x, y, int(text), ha='center', va='center', 
                    color='white',
                    fontsize='x-small',
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])

def show(ax, t, label=False):
    t = T.tensor(t) if not isinstance(t, T.Tensor) else t
    t = np.ma.masked_where(t == 0, t)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.imshow(t, vmin=0, vmax=9)
    if label:
        label_cells(ax, t)

def output(fname=None, show=True):
    if show: 
        plt.show()
    if fname is not None:
        print(f'Saving figure to {fname}...')
        plt.gcf().set_size_inches(WIDTH, HEIGHT)
        plt.subplots_adjust(wspace=-0.55, hspace=0.1)
        # plt.gcf().tight_layout()
        plt.savefig(fname, bbox_inches='tight')
    plt.close()

def viz_json(fname, save=False):
    data = json.load(open(fname, 'r'))
    name = clean(fname)
    xs = [d['input'] for d in data['train']]
    ys = [d['output'] for d in data['train']]
    viz_sample(xs, ys, title=f'pdfs/{name}.pdf')

def viz_sample(xs, ys, title='', text='', save=False):
    """Visualize an nx2 grid of input/output pairs"""
    f, ax = plt.subplots(len(xs), 2)
    plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=9)
    if len(xs) == 1:
        show(ax[0], xs[0])
        show(ax[1], ys[0])
    else:
        for i, (x, y) in enumerate(zip(xs, ys)):
            show(ax[i, 0], x)
            show(ax[i, 1], y)
    output()

def viz_mult(ts, text=''):
    f, ax = plt.subplots(1, len(ts))
    plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=9)
    for i, t in enumerate(ts):
        show(ax[i], t)
    output()

def viz(t, title='', subtitle='', fname=None, label=False):
    # f, ax = plt.plot() # subplots(2, 1)
    f, ax = plt.gcf(), plt.gca()
    plt.title(title)
    plt.figtext(0.5, 0.01, subtitle, wrap=True, horizontalalignment='center', fontsize=9)
    show(ax, t, label=label)
    output(fname, show=True)

def viz_grid(grid, text=''):
    """Visualize a grid of samples"""
    h, w = grid.shape[:2]
    f, ax = plt.subplots(w, h)
    plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=9)
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            show(ax[c, r], cell)
    output()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: viz.py file.json")
    else:
        fname = sys.argv[1]
        viz_json(fname)


