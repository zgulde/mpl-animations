from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rc('figure', figsize=(16, 9))

xlim = (-6, 6)
fig, ax = plt.subplots()
mu = [*np.linspace(0, 2, 100), *np.linspace(2, -2, 200), *np.linspace(-2, 0, 100),
      *np.repeat(0, 400)]
sigma = [*np.repeat(1, 400), *np.linspace(1, .6, 150), *np.linspace(.6, 3, 150),
         *np.linspace(3, 1, 100)]
x = np.linspace(-6, 6, 200)
def animate(i):
    mu_i = mu[i]
    sigma_i = sigma[i]
    ax.clear()
    if i in range(400):
        txt = 'Changing $\mu$ moves the distribution left and right\n'
        txt+= 'Increasing $\mu$ moves to the right\n'
        txt+= 'Decreasing $\mu$ moves to the left\n'
        ax.text(-5.5, .65, txt, va='top')
    elif i in range(400, 799):
        txt = 'Changing $\sigma$ changes how "spread out" the distribution is\n'
        txt+= 'Decreasing $\sigma$ makes it more narrow\n'
        txt+= 'Increasing $\sigma$ makes it more wide\n'
        ax.text(-5.5, .65, txt, va='top')
    title = 'The Normal Distribution\n'
    title+= f'$\mu$ = {mu_i:.2f}\n$\sigma$ = {sigma_i:.2f}'
    ax.set(title=title, xlim=xlim, ylim=(0, .7), ylabel='P(X = x)', xlabel='$x$')
    y = stats.norm(mu_i, sigma_i).pdf(x)
    ax.plot(x, y, c='firebrick')
anim = FuncAnimation(fig, animate, interval=16, frames=range(len(mu)), repeat=False)

fp = 'normal-distribution-demo.mp4'
print(f'Saving animation to {fp}')
anim.save(fp)
