# How do we make the animation from one year to another "smooth"? We need to add
# a bunch of points in between.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from vega_datasets import data

# https://stackoverflow.com/questions/45846765/efficient-way-to-unnest-explode-multiple-list-columns-in-a-pandas-dataframe
def explode(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
def pad_df(df, cols, n_points=100):
    # nuke indexes? TODO: test this w/ a df w/ datetime idx
    df = df.copy()
    df[cols] = df[cols].apply(_pad_series, n_points=n_points)
    return explode(df, cols)

# by_month = df.groupby(df.date.dt.strftime('%Y-%m')).mean().reset_index()
# by_month
# pad_df(by_month, ['temp'], 4)

df = data.seattle_weather()
df = df.assign(temp_avg=(df.temp_min + df.temp_max) / 2)
xtab = pd.crosstab(df.date.dt.strftime('%Y-%m'), df.weather).reset_index()
df = pad_df(xtab, list(xtab.columns[1:]), 30).set_index('date')

sf_temps = data.sf_temps().set_index('date').resample('D').mean().reset_index()
sf_temps['temp_bin'] = pd.qcut(sf_temps.temp, 4, labels=['cold', 'cool', 'warm', 'hot'])
sf_temps['temp_bin_desc'] = pd.qcut(sf_temps.temp, 4).apply(lambda i: f' ({i.left:.0f} - {i.right:.0f})')
sf_temps.temp_bin = sf_temps.temp_bin.astype(str) + sf_temps.temp_bin_desc.astype(str)
xtab = pd.crosstab(sf_temps.date.dt.strftime('%Y-%m'), sf_temps.temp_bin)
xtab.columns = list(xtab.columns)
xtab = xtab.reset_index()
df = pad_df(xtab, list(xtab.columns[1:]), 60)
df.set_index(pd.to_datetime(df.date).dt.strftime('%B %Y'), inplace=True)
df.drop(columns='date', inplace=True)

colors = ['lightblue', 'blue', 'red', 'orange']
fig, ax = plt.subplots()
def animate(i):
    ax.clear()
    ax.set(xlim=(0, 31), title=df.iloc[i].name)
    # ax.tick_params(rotation=0)
    df.iloc[i].plot.barh(ax=ax, width=1, color=colors)
anim = FuncAnimation(fig, animate, frames=range(len(df)), interval=32)
plt.show()
