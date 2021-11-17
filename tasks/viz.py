import json
import sys
import torch as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

WIDTH=10
HEIGHT=10

def clean(fname):
    name = '.'.join(fname.split('.')[:-1])
    return '/'.join(name.split('/')[1:])

def label(ax, grid):
    for (y,x), text in np.ndenumerate(grid):
        if text != 0:
            ax.text(x, y, text, ha='center', va='center', 
                    color='white',
                    fontsize='x-small',
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: viz.py file.json")
    else:
        fname = sys.argv[1]
        data = json.load(open(fname, 'r'))
        name = clean(fname)
        print(name)
        for i, d in enumerate(data['train']):
            x, y = d['input'], d['output']
            x, y = T.tensor(x), T.tensor(y)

            vmin=min(x.min().item(), y.min().item())
            vmax=max(x.max().item(), y.max().item())

            f, ax = plt.subplots(2,2) 
            f.suptitle(f'{name}-{i}')
            ax[0, 0].set_title('in')
            ax[0, 0].imshow(x, vmin=vmin, vmax=vmax)
            ax[0, 1].set_title('out')
            ax[0, 1].imshow(y, vmin=vmin, vmax=vmax)

            ax[1, 0].imshow(x, vmin=vmin, vmax=vmax)
            ax[1, 1].imshow(y, vmin=vmin, vmax=vmax)
            label(ax[1, 0], x)
            label(ax[1, 1], y)

            # plt.show()
            plt.gcf().set_size_inches(WIDTH, HEIGHT)
            plt.savefig(f'images/{name}-{i}.png')

