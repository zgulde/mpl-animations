# +
import numpy as np
import pandas as pd
import sklearn.preprocessing
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from IPython.display import HTML
import scipy.stats as stats

from functools import partial
import pytweening
import tween
# -

plt.ioff()
# https://stackoverflow.com/questions/43445103/inline-animations-in-jupyter
# plt.rcParams["animation.html"] = "jshtml"
# http://louistiao.me/posts/notebooks/save-matplotlib-animations-as-gifs/
plt.rc('animation', html='html5')

# +
fig, ax = plt.subplots()
x = np.random.normal(10, 2, 100)

x.sort()
lines, = ax.plot(x, marker='.', ls='')

plt.show()
lines

# +
fig, ax = plt.subplots()
x = np.random.normal(10, 2, 100)
x.sort()

lines, = ax.plot([], ls='', marker='.')
ax.set(xlim=(0, 100), ylim=(4, 16))

def animate(i):
    lines.set_data(range(len(x[:i])), x[:i])

anim = FuncAnimation(fig, animate, interval=1000 / 30, frames=len(x))
anim

# +
N = 48

x1 = np.linspace(-3, 3)
x2 = x1
xx = np.linspace(x1, x2, N)

y1 = np.zeros(50)
y2 = x2 ** 2
yy = np.linspace(y1, y2, N)

fig, ax = plt.subplots()

lines, = ax.plot([], [], marker='.')
ax.set(ylim=(yy.min(), yy.max()), xlim=(xx.min(), xx.max()))

def animate(i):
    x, y = xx[i], yy[i]
    lines.set_data(x, y)
    
anim = FuncAnimation(fig, animate, interval=1000 / 24, frames=N, repeat=False)
anim
# -

from scipy import stats

# +
x = np.linspace(-3.5, 3.5, 50)
y = stats.norm().pdf(x)

fig, ax = plt.subplots()

ax.plot(x, y)
poly = ax.fill_between([], [])

def animate(i):
    global poly
    poly.remove()
    poly = ax.fill_between(x[:i], y[:i], color='black')

anim = FuncAnimation(fig, animate, interval=1000/50, frames=50)
anim
# + {}
fps = 24

plt.rc('axes.spines', top=False, right=False)
plt.rc('font', size=16)
plt.rc('axes', grid=False)

x = np.random.uniform(2, 4, fps)
y = x + np.random.normal(.5, .5, fps)
x2 = sklearn.preprocessing.MinMaxScaler().fit_transform(x.reshape(-1, 1)).ravel()
xx = np.linspace(x, x2, fps)
y2 = sklearn.preprocessing.MinMaxScaler().fit_transform(y.reshape(-1, 1)).ravel()
yy = np.linspace(y, y2, fps)

fig, ax = plt.subplots(figsize=(16, 9))
lines, = ax.plot([], [], marker='o', ls='', color='firebrick')

ax.set(xticks=range(-1, 5), yticks=range(-1, 6))
ax.set(ylim=(y.min() - .3, y.max() + .3), xlim=(1.7, 4.3))

def animate(i):
    if i < fps * 1:
        ax.set(title='Original Data')
        lines.set_data(x[:i], y[:i])
    elif i < fps * 2:
        ax.set(title='Scale the X values')
        xmin = np.linspace(1.7, -.3, fps)
        ax.set(xlim=(xmin[i % fps], 4.3))
    elif i < fps * 3:
        lines.set_data(xx[i % fps], y)
    elif i < fps * 4:
        xmax = np.linspace(4.3, 1.3, fps)
        ax.set(xlim=(-.3, xmax[i % fps]))
    elif i < fps * 5:
        ax.set(title='Scale the Y values')
        ymin = np.linspace(y.min() - .3, -.3, fps)
        ax.set(ylim=(ymin[i % fps], y.max() + .3))
    elif i < fps * 6:
        lines.set_data(x2, yy[i % fps])
    elif i < fps * 7:
        ymax = np.linspace(y.max() + .3, 1.3, fps)
        ax.set(ylim=(-.3, ymax[i % fps]))

anim = FuncAnimation(fig, animate, interval=1000/fps, frames=fps * 7, repeat=True, repeat_delay=1000)
anim.save('min-max-scaling.mp4')
anim

# +
fig, ax = plt.subplots()
my_text = "Hello there!"
ax.set(xlim=(-1, 1), ylim=(-1, 1))
text = ax.text(0, 0, my_text[0], ha="left", va="center", size=16)

def animate(i):
    text.set_text(my_text[:i])

anim = FuncAnimation(
    fig, animate, interval=300, frames=range(len(my_text)), repeat=True
)
HTML(anim.to_html5_video())

# +
fps = 24

t = np.linspace(0,2*np.pi)
x = np.sin(t)

fig, ax = plt.subplots()
ax.axis([0,2*np.pi,-1,1])
l, = ax.plot([],[])

def animate(i):
    l.set_data(t[:i], x[:i])

ani = FuncAnimation(fig, animate, interval=1000 / fps, frames=len(t))

HTML(ani.to_html5_video())

# +
fps = 24

# x = np.random.uniform(2, 4, fps)
# x = np.random.normal(3, .5, fps)
x = stats.skewnorm(5, 2.5, .5).rvs(fps)
y = x + np.random.normal(.5, .5, fps)
x2 = sklearn.preprocessing.QuantileTransformer().fit_transform(x.reshape(-1, 1)).ravel()
xx = np.linspace(x, x2, fps)
y2 = sklearn.preprocessing.QuantileTransformer().fit_transform(y.reshape(-1, 1)).ravel()
yy = np.linspace(y, y2, fps)

fig, ax = plt.subplots()
lines, = ax.plot([], [], marker='o', ls='', color='firebrick')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set(xticks=range(-1, 5), yticks=range(-1, 6))
ax.set(ylim=(y.min() - .3, y.max() + .3), xlim=(1.7, 4.3))

def animate(i):
    if i < fps * 1:
        ax.set(title='Original Data')
        lines.set_data(x[:i], y[:i])
    elif i < fps * 2:
        ax.set(title='Scale the X values')
        xmin = np.linspace(1.7, -.3, fps)
        ax.set(xlim=(xmin[i % fps], 4.3))
    elif i < fps * 3:
        lines.set_data(xx[i % fps], y)
    elif i < fps * 4:
        xmax = np.linspace(4.3, 1.3, fps)
        ax.set(xlim=(-.3, xmax[i % fps]))
    elif i < fps * 5:
        ax.set(title='Scale the Y values')
        ymin = np.linspace(y.min() - .3, -.3, fps)
        ax.set(ylim=(ymin[i % fps], y.max() + .3))
    elif i < fps * 6:
        lines.set_data(x2, yy[i % fps])
    elif i < fps * 7:
        ymax = np.linspace(y.max() + .3, 1.3, fps)
        ax.set(ylim=(-.3, ymax[i % fps]))

anim = FuncAnimation(fig, animate, interval=1000/fps, frames=fps * 7, repeat=True, repeat_delay=1000)
anim.save('quantile-scaling.mp4')
anim

# +
from pytweening import easeOutCubic, easeInCubic
import tween

tween.tween(10, 1, easeInCubic, 24)

# +
fps = 24

plt.rc('axes.spines', top=False, right=False)
plt.rc('font', size=16)
plt.rc('axes', grid=False)

x = np.random.uniform(2, 4, fps)
y = x + np.random.normal(.5, .5, fps)
x2 = sklearn.preprocessing.MinMaxScaler().fit_transform(x.reshape(-1, 1)).ravel()
# xx = np.linspace(x, x2, fps)
xx = tween(x, x2, easeInCubic, fps)
y2 = sklearn.preprocessing.MinMaxScaler().fit_transform(y.reshape(-1, 1)).ravel()
# yy = np.linspace(y, y2, fps)
yy = tween(y, y2, easeInCubic, fps)

fig, ax = plt.subplots(figsize=(16, 9))
lines, = ax.plot([], [], marker='o', ls='', color='firebrick')

ax.set(xticks=range(-1, 5), yticks=range(-1, 6))
ax.set(ylim=(y.min() - .3, y.max() + .3), xlim=(1.7, 4.3))

def animate(i):
    if i < fps * 1:
        ax.set(title='Original Data')
        lines.set_data(x[:i], y[:i])
    elif i < fps * 2:
        pass # pause
    elif i < fps * 3:
        ax.set(title='Min-Max Scale to 0-1')
        # expand x and y axis
        xmin = np.linspace(1.7, -.3, fps)
        xmin = tween(1.7, -.3, easeInCubic, fps)
        ymin = tween(y.min() - .3, -.3, easeInCubic, fps)
        ax.set(xlim=(xmin[i % fps], 4.3), ylim=(ymin[i % fps], y.max() + .3))
    elif i < fps * 4:
        pass # pause
    elif i < fps * 5:
        ax.set(title='scale the x values')
        # scale x values
        lines.set_data(xx[i % fps], y)
    elif i < fps * 6:
        pass # pause
    elif i < fps * 7:
        ax.set(title='scale the y values')
        # scale y values
        lines.set_data(x2, yy[i % fps])
    elif i < fps * 8:
        pass # pause
    elif i < fps * 9:
        ax.set(title='Same shape as the original data, but the scale has changed')
        # shrink x and y axis
        xmax = tween(4.3, 1.3, easeInCubic, fps)
        ymax = tween(y.max() + .3, 1.3, easeInCubic, fps)
        ax.set(xlim=(-.3, xmax[i % fps]), ylim=(-.3, ymax[i % fps]))

anim = FuncAnimation(fig, animate, interval=1000/fps, frames=fps * 9, repeat=True, repeat_delay=1000)
anim.save('min-max-scaling.mp4')
anim


# +
fps = 12

import tween
tween = partial(tween.tween, n_points=fps, fn=pytweening.easeInSine)

# +
np.random.seed(123)

x = np.random.normal(5, 2, 100)
x_centered = x - x.mean()
x_scaled = x_centered / x.std()
xx_centered = tween(x, x_centered)
xx_scaled = tween(x_centered, x_scaled)

y = x + np.random.uniform(-1, 1, 100)
y_centered = y - y.mean()
y_scaled = y_centered / y.std()
yy_centered = tween(y, y_centered)
yy_scaled = tween(y_centered, y_scaled)


fig, ax = plt.subplots()

lines, = ax.plot(x, y, ls='', marker='.')
ybarline, = ax.plot([y.mean(), y.mean()], [x.min(), x.max()], ls=':', color='grey')
xbarline, = ax.plot([y.min(), y.max()], [x.mean(), x.mean()], ls=':', color='grey')

def animate(i):
    if i < fps * 1:
        xmin = tween(x.min(), x_centered.min() - .5)
        ymin = tween(y.min(), y_centered.min() - .5)
        ax.set(xlim=(xmin[i%fps], ax.get_xlim()[1]), ylim=(ymin[i%fps], ax.get_ylim()[1]))
    elif i < fps * 2:
        pass
    elif i < fps * 3:
        ybar_i = tween(y.mean(), 0)[i % fps]
        ymin_i = tween(y.min(), -4)[i % fps]
        ymax_i = tween(y.max(), 4)[i % fps]
        xbar_i = tween(x.mean(), 0)[i % fps]
        xmin_i = tween(x.min(), -4)[i % fps]
        xmax_i = tween(x.max(), 4)[i % fps]
        ybarline.set_data([ybar_i, ybar_i], [xmin_i, xmax_i])
        xbarline.set_data([ymin_i, ymax_i], [xbar_i, xbar_i])
    elif i < fps * 4:
        pass
    elif i < fps * 5:
        lines.set_data(xx_centered[i % fps], yy_centered[i % fps])
    elif i < fps * 6:
        pass
    elif i < fps * 7:
        lines.set_data(xx_scaled[i % fps], yy_scaled[i % fps])
    elif i < fps * 8:
        pass

anim = FuncAnimation(fig, animate, interval=1000/fps, frames=fps * 8, repeat=True)
anim
# -

plt.close()


def make_animation(fig, fps, fns, **kwargs):
    def animate(i):
        fn = fns[i // fps]
        if fn is not None: fn(i % fps)

    return FuncAnimation(fig, animate, interval=1000/fps, frames=fps * len(fns), repeat=True, **kwargs)


# +
import tween

fps = 24
tween = partial(tween.tween, n_points=fps, fn=pytweening.easeInSine)

fig, ax = plt.subplots()
lines, = ax.plot([], [], ls='', marker='o')

n = 100

x1 = np.random.uniform(1, 5, n)
x2 = np.random.uniform(1, 5, n)
xx = tween(x1, x2)

y1 = np.random.uniform(1, 5, n)
y2 = np.random.uniform(1, 5, n)
yy = tween(y1, y2)

ax.set(xlim=(.5, 5.5), ylim=(.5, 5.5))

def show_points(artist, x, y, fps):
    def fn(i):
        n = x1.shape[0]
        i = round(i * (n / fps))
        artist.set_data(x1[:i], y1[:i])
    return fn
    
def transform(i):
    lines.set_data(xx[i], yy[i])
    
make_animation(fig, fps, [
    show_points(lines, x1, y2, fps),
    None,
    transform,
])

# +
fig, ax = plt.subplots()
lines, = ax.plot([], [], marker='.', c='black', ls='')

anim = Animation(24, fig, lines=lines)

x1 = np.random.uniform(0, 1, 100)
x2 = np.random.uniform(0, 1, 100)

y1 = np.random.uniform(0, 1, 100)
y2 = np.random.uniform(0, 1, 100)

anim.show_points(x1, y1)
anim.pause()
anim.transform(x1, x1, y1, y2)
anim.transform(x1, x2, y2, y2)

anim.animate()

# +
fig, ax = plt.subplots()
lines, = ax.plot([], [], marker='.', c='black', ls='')

anim = Animation(24, fig, lines=lines)

y = np.random.uniform(0, .6, 100)
y.sort()
y2 = y + np.linspace(.4, 0, 100)

x = np.arange(y.size) / 100

anim.show_points(x, y)
anim.transform(x, x, y, y2)
anim.pause()


anim.animate()

# +
from matplotlib.animation import FuncAnimation
from tween import tween
import pytweening

class Animation:
    def __init__(self, fps, fig, lines=None, ax=None, timing_fn=None):
        self._fps = fps
        self._fig = fig
        self._fns = []
        self._lines = lines
        self._ax = ax
        self._timing_fn = timing_fn or pytweening.easeInQuad

    def __call__(self, fn):
        self._fns.append(fn)
        return fn
    
    def pause(self):
        self._fns.append(None)

    def show_points(self, x, y, lines=None):
        if lines is None:
            lines = self._lines
        def fn(i):
            n = x.shape[0]
            i = round(i * (n / self._fps))
            lines.set_data(x[:i], y[:i])
        self._fns.append(fn)
        
    def transform(self, x1, x2, y1, y2, lines=None):
        if lines is None:
            lines = self._lines
        xx = tween(x1, x2, self._timing_fn, fps)
        yy = tween(y1, y2, self._timing_fn, fps)
        def fn(i):
            lines.set_data(xx[i], yy[i])
        self._fns.append(fn)
    
    def animate(self):
        if len(self._fns) == 0:
            raise Exception("No functions to animate!")
        def _animate(i):
            fn = self._fns[i // self._fps]
            if fn is not None:
                fn(i % self._fps)

        return FuncAnimation(
            self._fig,
            _animate,
            interval=1000 / self._fps,
            frames=len(self._fns) * self._fps,
            repeat=True,
        )


# -

plt.rc('font', size=14)

# +
from tween import tween

fps = 24
mytween = partial(tween, n_points=fps, fn=pytweening.easeOutQuad)

# +
np.random.seed(123)

x = np.random.normal(6, 2, 50)
x2 = x - x.mean()
x3 = x2 / x2.std()

y = np.random.normal(6, 2, 50)
y2 = y - y.mean()
y3 = y2 / y2.std()

fig = plt.figure(figsize=(16, 9))
ax = fig.add_axes((.05, .05, .9, .8))
lines, = ax.plot([], [], marker='.', ls='')
ax.set(ylim=(y.min() - .5, y.max() + .5), xlim=(x.min() - .5, x.max() + .5))
ax.set(title='Centering and Scaling', ylabel='$y$', xlabel='$x$')
ax.hlines(0, x2.min() - .5, x.max() + .5, ls=':', color='grey')
ax.vlines(0, y2.min() - .5, y.max() + .5, ls=':', color='grey')

anim = Animation(fps=fps, fig=fig, lines=lines)

anim.show_points(x, y)
anim.pause()

@anim
def scale_xaxis(i):
    if i == 0:
        ax.set(title="Center x by subtracting the mean\n$x' = x - \\mu_x$")
    xx = mytween(ax.get_xlim()[0], x2.min() - .5)
    ax.set(xlim=(xx[i], ax.get_xlim()[1]))

anim.transform(x, x2, y, y)
anim.pause()

@anim
def scale_yaxis(i):
    if i == 0:
        ax.set(title="Center y by subtracting the mean\n$y' = y - \\mu_y$")
    yy = mytween(ax.get_ylim()[0], y2.min() - .5)
    ax.set(ylim=(yy[i], ax.get_ylim()[1]))

anim.transform(x2, x2, y, y2)
anim.pause()

@anim
def pre_scale(i):
    ax.set(title="Scale the data by dividing by the standard deviation\n$x' = x / \\sigma_x$\n$y' = y / \\sigma_y$")

anim.transform(x2, x3, y2, y3)
anim.pause()

@anim
def scale_xaxis_and_yaxis(i):
    if i == 0:
        ax.set(title="The shape of the data is the same as the orignal, but with\n$\mu_x = \mu_y = 0$ and\n$\sigma_x = \sigma_y = 1$")
    xmin = mytween(ax.get_xlim()[0], -3.5)
    xmax = mytween(ax.get_xlim()[1], 3.5)
    ymin = mytween(ax.get_ylim()[0], -3.5)
    ymax = mytween(ax.get_ylim()[1], 3.5)
    ax.set(xlim=(xmin[i], xmax[i]), ylim=(ymin[i], ymax[i]))

anim.pause()
anim.animate().save('center-and-scale.mp4')
