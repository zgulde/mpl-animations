import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

fp = './regression-evaluation.html'
print(f'saving to {fp}')
with open(fp, 'w+') as f:
    f.write(anim.to_html5_video())

