# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 21:33:46 2020

@author: raifo
"""

import pandas as pd
from stldecompose import decompose
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
sns.set_style('darkgrid')

data = pd.read_csv('testing_results/test_result.csv')
data = data.loc[data['StreamSegmentId'] == 62]

temp = data.reset_index()
df_stl_minute = temp[['StartDateTime', 'AObs']].set_index('StartDateTime')
decomp = decompose(df_stl_minute.values, period=12 )

ts = pd.Series(df_stl_minute['AObs'])
trend = pd.Series(decomp.trend)
seasonal = pd.Series(decomp.seasonal)
residual = pd.Series(decomp.resid)



res = seasonal_decompose(df_stl_minute, model = 'additive', freq = 12)

res.plot()
plt.show()
 

plt.subplot(4,1,1)
ts.plot()
plt.subplot(4,1,2)
seasonal.plot()
plt.subplot(4,1,3)
trend.plot()
plt.subplot(4,1,4)
residual.plot()
plt.show()








