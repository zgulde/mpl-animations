from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rc('figure', figsize=(16, 9))

p = [*np.repeat(.5, 50), *np.linspace(.5, 1, 50), *np.linspace(1, 0, 100)]
n = [*np.linspace(5, 25, 50).round(), *np.repeat(25, 150)]
fig, ax = plt.subplots()
x = np.arange(0, 26)
def animate(i):
    p_i = p[i]
    n_i = n[i]
    ax.clear()
    if i in range(50):
        ax.text(0, .3, 'As we increase $n$, we would expect to see more successes')
    elif i in range(50, 100):
        ax.text(0, .3, 'Increasing $p$ also increases the number of successes')
    elif i in range(120, 199):
        ax.text(25, .3, 'Decreasing p decreases the number of successes', ha='right')
    elif i == 199:
        ax.text(25, .3, 'When p is 0, the only possible outcome is 0 successes', ha='right')
    title = 'The Binomial Distribution\n'
    title+= f'n, number of trials, = {n_i}\np, P(success), = {p_i:.2f}'
    ax.set(title=title, ylim=(0, .33), xlabel='$x$, the number of "successes"', ylabel='P(X = x)')
    y = stats.binom(n_i, p_i).pmf(x)
    ax.bar(x, y, color='lightblue', width=1, edgecolor='black')
anim = FuncAnimation(fig, animate, interval=100, frames=range(200), repeat=False)
plt.show()

fp = './binomial-distribution-demo.html'
print(f'Saving animation to {fp}')
with open(fp, 'w+') as f:
    f.write(anim.to_html5_video())
