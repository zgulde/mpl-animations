from zgulde.ds_imports import *
from matplotlib.animation import FuncAnimation
plt.rc('figure', figsize=(14, 10))
plt.ioff()

fig, ax = plt.subplots()
my_text = 'Hello there!'
ax.set(xlim=(-1, 1), ylim=(-1, 1))
text = ax.text(0, 0, my_text[0], ha='left', va='center', size=16)
def animate(i):
    text.set_text(my_text[:i])
anim = FuncAnimation(fig, animate, repeat=True)
plt.show()

###

fig, ax = plt.subplots()
x = np.linspace(0, 1)
y = x ** 2
scatter = ax.scatter(x[0], y[0])
def animate(i):
    scatter.set_offsets(x[:i], y[:i])
anim = FuncAnimation(fig, animate, repeat=True)
plt.show()

###

fig, ax = plt.subplots()
y = np.random.randn(500)
ax.hist(y[0])
ax.set(xlim=(-4, 4), ylim=(0, 75))
def animate(i):
    ax.clear()
    ax.set(xlim=(-4, 4), ylim=(0, 75))
    ax.hist(y[:i])
    ax.set_title(f'n = {i}')
anim = FuncAnimation(fig, animate, repeat=True, interval=30)
plt.show()

###

from vega_datasets import data

df = data.seattle_weather()
df = (df.set_index('date')
 [['temp_min', 'temp_max']]
 .assign(temp_min=lambda df: df.temp_min * 9/5 + 32,
         temp_max=lambda df: df.temp_max * 9/5 + 32))
fig, ax = plt.subplots()
xlim, ylim = (0, 2), (df.temp_min.min(), df.temp_max.max())
df.iloc[0].plot.bar(ax=ax, title=df.index[0].strftime('%Y-%m-%d'))
ax.set(xlim=xlim, ylim=ylim)
def animate(i):
    ax.clear()
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_title(df.index[i].strftime('%Y-%m-%d'))
    df.iloc[i].plot.bar(ax=ax)
anim = FuncAnimation(fig, animate, repeat=True, frames=range(len(df)), interval=30)
plt.show()

###

x = np.linspace(-4, 4, 50)
y = x**2
fig, ax = plt.subplots()
ax.set(xlim=(-4, 4))
line = ax.plot(x[::10], y[::10], 'k', marker='o')[0]
def animate(i):
    line.set_xdata(x[::i])
    line.set_ydata(y[::i])
anim = FuncAnimation(fig, animate, frames=[20, 15, 10, 9, 8, 7, 6, 3, 1])
plt.show()

###

fig, ax = plt.subplots()
sums = []
bins = range(1, 14)
def roll(ndice=1, nsides=6):
    return np.random.randint(1, nsides+1, ndice)
def plot():
    rolls = roll(2)
    sums.append(rolls.sum())
    ax.set(xlim=(1, 13), xticks=range(1, 14), ylim=(0, 100))
    ax.set_title(str(len(sums)) + ' pairs of dice rolled')
    ax.hist(sums, bins=bins, align='left')
plot()
def animate(i):
    ax.clear()
    return plot()
anim = FuncAnimation(fig, animate, frames=range(10), interval=30)
plt.show()

###

n_dice = 10
bins = np.arange(1, 6, .1)
xlim = (0, 7)
xticks = [1, 2, 2.5, 3, 3.5, 4, 5]
fig, ax = plt.subplots()
means = []
def roll(ndice=1, nsides=6):
    return np.random.randint(1, nsides+1, ndice)
def plot():
    rolls = roll(n_dice)
    means.append(rolls.mean())
    ax.set(xlim=xlim, xticks=xticks, ylim=(0, 40))
    ax.set_title(f'{len(means):,} averages of {n_dice} dice rolls')
    # ax.set_title(f'{rolls} = {rolls.mean():.1f}')
    ax.hist(means, bins=bins, align='left')
plot()
def animate(i):
    print(i)
    ax.clear()
    return plot()
anim = FuncAnimation(fig, animate, frames=range(10), interval=30)
plt.show()

###

from sklearn.linear_model import LinearRegression
from pydataset import data
import textwrap as tw

fig, ax = plt.subplots()
ax.set(xlim=(0, 45), ylim=(0, 8))
tips = data('tips').sample(8, random_state=456)
lm = LinearRegression().fit(tips[['total_bill']], tips.tip)
m = lm.coef_[0]
b = lm.intercept_
x = tips.total_bill
y = tips.tip
yhat = m * x + b
def animate(i):
    if i == 0:
        ax.clear()
        ax.plot(x, y, ls='', marker='o', label='actual ($y$)', color='black')
        ax.set(xlim=(0, 45), ylim=(0, 8), title='Our orignal X and Y data')
        ax.legend(loc='upper left')
    elif i == 1:
        ax.set(title='Add our line to make predictions, y = {:.2f}x + {:.2f}'.format(m, b))
        ax.plot(x, yhat, color='grey', label='')
        ax.plot(x, yhat, label='predicted ($\hat{y}$)', marker='x', ls='',
                color='darkgreen', markersize=11)
        ax.legend(loc='upper left')
    elif i == 2:
        ax.set(title='The residuals are the distance between the predicted and actual value')
        ax.vlines(x, yhat, y, linestyle=':', label='residuals ($y - \hat{y}$)', color='red')
        ax.legend(loc='upper left')
    elif i == 3:
        for xi, yi, yhati in zip(x, y, yhat):
            ax.text(xi + .5, (yi + yhati) / 2, '{:.2f}'.format(yi - yhati), ha='right')
    elif i == 4:
        ax.set(title='Evaluation metrics are derived from the residuals')
        sse = ((y - yhat) ** 2).sum()
        t = tw.dedent('''
        Sum of Squred Errors is the sum of the residuals, squared
        $SSE = \sum(y - \hat{y}) = %.2f$
        ''' % sse).strip()
        ax.text(2.5, 6, t)
    elif i == 5:
        sse = ((y - yhat) ** 2).sum()
        mse = sse / tips.shape[0]
        t = tw.dedent(r'''
        Mean Squared Error is the SSE divided by the number of points
        $MSE = \frac{\sum(y - \hat{y})}{n} = %.2f$
        ''' % mse).strip()
        ax.text(2.5, 5.25, t)
    elif i == 6:
        sse = ((y - yhat) ** 2).sum()
        mse = sse / tips.shape[0]
        rmse = math.sqrt(mse)
        t = tw.dedent(r'''
        Root Mean Squared Error is the square root of the MSE
        RMSE has the same units as our y variable
        $RMSE = \frac{\sum(y - \hat{y})}{n} = %.2f$
        ''' % rmse).strip()
        ax.text(2.5, 4.3, t)
anim = FuncAnimation(fig, animate, interval=2000, frames=range(7), repeat=False)
plt.show()
with open('./regression-evaluation.html', 'w+') as f:
    f.write(anim.to_html5_video())

###

xlim = (-6, 6)
fig, ax = plt.subplots()
mu = [*np.linspace(0, 2, 50), *np.linspace(2, -2, 100), *np.linspace(-2, 0, 50),
      *np.repeat(0, 200)]
sigma = [*np.repeat(1, 200), *np.linspace(1, .6, 75), *np.linspace(.6, 3, 75),
         *np.linspace(3, 1, 50)]
x = np.linspace(-6, 6, 200)
def animate(i):
    mu_i = mu[i]
    sigma_i = sigma[i]
    ax.clear()
    if i in range(200):
        txt = 'Changing $\mu$ moves the distribution left and right\n'
        txt+= 'Increasing $\mu$ moves to the right\n'
        txt+= 'Decreasing $\mu$ moves to the left\n'
        ax.text(-5.5, .65, txt, va='top')
    elif i in range(200, 399):
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
plt.show()

with open('./normal-distribution-demo.html', 'w+') as f:
    f.write(anim.to_html5_video())

###


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

with open('./binomial-distribution-demo.html', 'w+') as f:
    f.write(anim.to_html5_video())
