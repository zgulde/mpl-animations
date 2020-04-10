import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

plt.rc("figure", figsize=(16, 9))

np.random.seed(123)
df = pd.DataFrame(dict(x=np.random.normal(0, 1, 1000))).assign(
    y=lambda df: df.x + np.random.normal(0, 0.5, 1000)
)
fig, ax = plt.subplots()
ymin, ymax = df.y.min(), df.y.max()
xmin, xmax = df.x.min(), df.x.max()
xticks = yticks = np.arange(-3, 4)


def animate(i):
    ax.clear()
    ax.set(ylim=(ymin, ymax), xlim=(xmin, xmax), xticks=xticks, yticks=yticks)
    data = df.iloc[:i]
    ax.scatter(data.x, data.y)


anim = FuncAnimation(fig, animate, interval=10, frames=range(1000), repeat=False)

plt.show()
