# How do we make the animation from one year to another "smooth"? We need to add
# a bunch of points in between.

from zgulde import and_next
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# https://stackoverflow.com/questions/45846765/efficient-way-to-unnest-explode-multiple-list-columns-in-a-pandas-dataframe
def explode(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]
def _pad_row(x, nextx, n_points):
    if not np.isnan(nextx):
        return np.linspace(x, nextx, n_points)
    else:
        return np.array([x])
def _pad_series(s, n_points):
    return [_pad_row(x, nextx, n_points) for x, nextx in zip(s.values, s.shift(-1).values)]
def pad_df(df, cols, n_points=100):
    # nuke indexes? TODO: test this w/ a df w/ datetime idx
    df = df.copy()
    df[cols] = df[cols].apply(_pad_series, n_points=n_points)
    return explode(df, cols)

from vega_datasets import data

# by_month = df.groupby(df.date.dt.strftime('%Y-%m')).mean().reset_index()
# by_month
# pad_df(by_month, ['temp'], 4)

df = data.seattle_weather()
df = df.assign(temp_avg=(df.temp_min + df.temp_max) / 2)
xtab = pd.crosstab(df.date.dt.strftime('%Y-%m'), df.weather).reset_index()
df = pad_df(xtab, list(xtab.columns[1:]), 15).set_index('date')

fig, ax = plt.subplots()
def animate(i):
    ax.clear()
    ax.set(xlim=(0, 25), title=df.iloc[i].name)
    df.iloc[i].plot.barh(ax=ax, width=.9)
anim = FuncAnimation(fig, animate, frames=range(len(df)), interval=32)
plt.show()
