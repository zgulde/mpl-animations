import pandas as pd
import numpy as np
from pydataset import data

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn.cluster import dbscan
from sklearn.preprocessing import scale

def add_circles(X, radius):
    for i, p in X.iterrows():
        circle = plt.Circle((p.hp, p.wt), radius, ec='black', fc='white', ls=':', zorder=-1, alpha=.4)
        plt.gca().add_patch(circle)

mtcars = data('mtcars')

plt.rc('axes', grid=True)
plt.rc('grid', linestyle=':', linewidth=.8, alpha=.7)
plt.rc('axes.spines', right=False, top=False)
plt.rc('figure', figsize=(11, 8))
plt.rc('font', size=12.0)
plt.rc('hist', bins=15)

eps = .8

X = pd.DataFrame(scale(mtcars[['hp', 'wt']]), columns=['hp', 'wt'])
cores, labels = dbscan(X, eps=eps, min_samples=3)
X['cluster'] = labels

X = X.sort_values(by='hp')

eps = .5

fig, ax = plt.subplots()
ax.set(xlim=(-3, 3), ylim=(-3, 3))

def animate(i):
    ax.clear()
    ax.set(xlim=(-3, 3), ylim=(-3, 3), title='dbscan, minPts=3, eps={}'.format(eps))
    ax.scatter(X.hp, X.wt, marker='.')
    d = X.iloc[0:i]
    add_circles(d, eps)
    for c in d.cluster.unique():
        subset = d[d.cluster == c]
        marker = 'o' if c >= 0 else 'x'
        ax.scatter(subset.hp, subset.wt, label=c, marker=marker)
    ax.legend()

anim = FuncAnimation(fig, animate, interval=300, frames=range(len(X) + 1), repeat=False)

fp = 'dbscan.mp4'
print(f'saving to {fp}')
anim.save(fp)
