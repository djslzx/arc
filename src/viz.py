import sys
import json
import seaborn as sns
import torch as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

WIDTH=10
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

def output(fname=None, show=True):
    if show: 
        plt.show()
    if fname is not None:
        print(f'Saving figure to {fname}...')
        plt.gcf().set_size_inches(WIDTH, HEIGHT)
        plt.savefig(fname)
    plt.close()

def viz_task(name, i, x, y):
    vmin=0 #min(x.min().item(), y.min().item())
    vmax=9 #max(x.max().item(), y.max().item())    
    x_masked = np.ma.masked_where(x == 0, x)
    y_masked = np.ma.masked_where(y == 0, y)

    f, ax = plt.subplots(2,2) 
    f.suptitle(f'{name}-{i}')
    ax[0, 0].set_title('in')
    ax[0, 0].imshow(x_masked, vmin=vmin, vmax=vmax)
    ax[0, 1].set_title('out')
    ax[0, 1].imshow(y_masked, vmin=vmin, vmax=vmax)

    ax[1, 0].imshow(x_masked, vmin=vmin, vmax=vmax)
    ax[1, 1].imshow(y_masked, vmin=vmin, vmax=vmax)
    label_cells(ax[1, 0], x)
    label_cells(ax[1, 1], y)

    output()
    # output(f'../tasks/images/{name}-{i}.png', show=False)

def viz_json(fname):
    data = json.load(open(fname, 'r'))
    name = clean(fname)
    for i, d in enumerate(data['train']):
        x, y = d['input'], d['output']
        x, y = T.tensor(x), T.tensor(y)
        viz_task(name, i, x, y)

def viz(t, title='', subtitle='', fname=None, show=True):
    f, ax = plt.subplots(2, 1)
    f.suptitle(title)
    plt.figtext(0.5, 0.01, subtitle, wrap=True, horizontalalignment='center', fontsize=9)
    t_masked = np.ma.masked_where(t == 0, t)
    ax[0].imshow(t_masked, vmin=0, vmax=9)
    ax[1].imshow(t_masked, vmin=0, vmax=9)
    # ax[0].grid()
    # ax[1].grid()
    label_cells(ax[1], t_masked)
    output(fname, show)

def viz_sample(samples, n, txt=''):
    '''Visualize an nxn grid of samples'''
    f, ax = plt.subplots(n, n)
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=9)    
    for i, sample in enumerate(samples):
        x, y = i%n, i//n
        masked = np.ma.masked_where(sample == 0, sample)
        ax[y, x].imshow(masked, vmin=0, vmax=9)
        ax[y, x].axes.xaxis.set_visible(False)
        ax[y, x].axes.yaxis.set_visible(False)
        # label_cells(ax[y,x], masked)
    output()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: viz.py file.json")
    else:
        fname = sys.argv[1]
        viz_json(fname)


