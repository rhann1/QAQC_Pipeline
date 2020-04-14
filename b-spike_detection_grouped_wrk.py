 # -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import datetime as dt
import json
from pandas.io.json import json_normalize, to_json
import pylab
import matplotlib.pyplot as plt

# Simple spiike detector.
# compares value with previous record and returns potential outlier if difference is > than %delta 
# only value above a given threshold are considered
def spike(x, mu, delta):
    df=x
    x_bar=x.mean()                       # store the mean of the windowed values
    if np.abs(df[0] - df[1]) >= delta:   # check if the difference between current and previous is > than threshold
        return 1
    else:
        return 0
    
# mean absoute deviation based spike detection.    
def spike1(x, mu, delta):
    x = x[::-1]
    df=x
    x_bar=np.abs(x).mean()
    if np.abs(df[0])> delta*x_bar : # if MAD is > assigned threshold then set flag.
        print(x, x_bar)
        return 1
    else:
        return 0
    
# modified z-score spike detector.  This is a robust method based on the median absolute deviation.
# at this point this is the preferable method for detection obvious outliers
def spike2(x, mzs_t):
    x = x[::-1]
    med_x = np.median(x) # find the median of the window
    mad = np.median(np.abs(np.subtract(x, -med_x))) # compute the median absolute deviation
    mzs = 0.675*(x[0] - med_x)/mad # compute the modified z-score (mzs)
    if mzs > 3.5:
#        print(x, med_x, mad, mzs)
        return 1
    else:
        return 0
    
# method to return msz for testing purposes and visualization (will be removed in production)
def mzs_test(x, mzs_t):
    x = x[::-1]
    med_x = np.median(x) # find the median of the window
    mad = np.median(np.abs(np.subtract(x, -med_x))) # compute the median absolute deviation
    mzs = 0.675*(x[0] - med_x)/mad # compute the modified z-score (mzs)
    return mzs


def spike3(x, zs_t):
    x = x[::-1]
    x_bar = x.mean()
    dev = np.abs(x[0]-x_bar)/x_bar
    if dev > 2 and x_bar > delta1:
        print(x)
        return 1
    else:
        return 0
    
def spike4(x, QA4, lowValue4):
    x = x[::-1]
    x_med = np.median(x)
    dev = np.abs(x[0] - x_med)/x_med
    if dev > QA4 and x_med > lowValue4:
        return 1
    else:
        return 0
    
    
def winCount(x):
    x = x[::-1]
    return len(x)
    
def spike3_mod(x, zs_t):
    x = x[::-1]
    x_bar = x[0:len(x)-2].mean()
    dev = np.abs(x[0]-x_bar)/x_bar
    if dev > 2 and x_bar > delta1:
#        print(x, x_bar, dev)
        return 1
    else:    
        return 0
    
def lowvar(x, p_delta):
    x = x[::-1]
    x_bar = x.mean()
    sdev = np.std(x)
#    print(np.abs(np.subtract(x,x_bar)).sum()/len(x))
    if np.abs(np.subtract(x,x_bar)).sum()/len(x) < p_delta*sdev:
        return 1
    else:
        return 0

# persistent value test.  This will check if the current observation has repeated over the length of the test window.    
def persist(x):
    x = x[::-1]
    sum_dev = sum([x[0]-x[i] for i in range(len(x))]) # computes the sum of the magnitudes of the deviation from x[0]. If the sum ==0, x[0] is a repeating value in the series. 
    if sum_dev == 0: # Check if observation is part of a repeating sequence.
        return 1
    else:
        return 0
           
def timeDiff(x, freq, tu):
    diff = np.abs(x[0] - x[1])
    #diff = diff/np.timedelta64(1,tu)
    if diff == freq:
        return 0
    if diff == 2*freq*1.0:
        return 1
    else:
        return 2

def timeDiff1(x, freq, tu):
    diff = (x[0] - x[-1])
    #diff = diff/np.timedelta64(1,tu)
    if diff == freq:
        return diff
    if diff == 2*freq*1.0:
        return diff
    else:
        return diff
    return x[0]

def t_delta(x):
    delta =( x - dt.datetime(1970,1,1) ).dt.total_seconds()
    delta = np.array(delta)/60
    return delta

def printx(x):
    return x[-1]
    
#statiscal parameter definitions
mu = 15
sigma = 7
freq = 60
tu = 'h'
window = 2
delta = 4
delta1 = 4
p_delta = 0.55

# ---------------------------------------------------------------------------------------------------------------------------------------------------
#generation of simulated sequence from a normal distribution testing QC functions and drivers
# ---------------------------------------------------------------------------------------------------------------------------------------------------
s = np.random.normal(mu, sigma, 8784)
s[4000] = 90
s1 = np.random.normal(mu, sigma, 8784)
s1[4400] = 275
s1[4370:4390] = 12.4
#generation of a simulated DateTime sequence
tidx = pd.date_range('2000-01-01', '2000-12-31 23:00', freq='H')

#creation of time series dataframe
data = pd.DataFrame({'date':tidx,'signal':s})
data1 = pd.DataFrame({'date':tidx,'signal':s1})
data1['date'].iloc[8780] = pd.to_datetime('2000-12-31 20:30:00')    #test row for timediff check
data1.drop(index=8774, inplace=True)

# initialize two group IDs
data['id']=0
data1['id']=1
# append ID=1 group
data = data.append(data1, ignore_index=True)

#data = pd.read_csv('OCAP_SJV.csv')
#data.rename(columns={'DateTime':'date', 'MeasuredValue':'signal', 'SiteName':'id'}, inplace=True)
data['date'] = pd.to_datetime(data['date'])
# --------------------------------------------------------------------------------------------------------------------------------------------------
# end simulated time-series generator
# --------------------------------------------------------------------------------------------------------------------------------------------------

# convert successive datetime objects to timedeltas on a unit seconds basis
#data = data.sort_values(['id','date'], ascending=False)
data['date1'] = pd.to_timedelta(data['date']).astype('timedelta64[m]').astype(int)

#prepare frame for group operation
data['date2'] = t_delta(data['date'])


frame = data
# --------------------------------------------------------------------------------------------------------------------------------------------------
# convert to JSON payload for JSON test
# this payload object would be retrieved from data API
# -------------------------------------------------------------------------------------------------------------------------------------------------
frame_json = json.loads(frame.to_json(orient = 'records', date_format = 'iso'))
# conversioin of the JSON payload to a dataframe
frame = json_normalize(frame_json)
# convert 'date' column to pandas 'datetime' object from JSON 'iso' format
frame['date'] = pd.to_datetime(frame['date'])
# --------------------------------------------------------------------------------------------------------------------------------------------------
# end JSON payload simulator
# --------------------------------------------------------------------------------------------------------------------------------------------------


# determine subhourly observations within the aggregation target time unit (standard: 1 hour for subhourly data)
hourCount = data.groupby('id').resample('D', on='date').count()

QA1 = 4
QA2 = 4.0
QA3 = 0.5
QA4 = 3.0
lowValue4 = 5

 
df_list = []

#data={"X":[1, 1, 2, 2, 3], "Y":[6, 7, 8, 9, 10], "Z":[11, 12, 13, 14, 15]}
data={"id":["A","A","A","B","B","B"], "X":[1, 1, 2, 2, 3, 4], "Y":[1, 1, 1, 4, 4, 4], "Z":[11, 12, 13, 14, 15, 4]}
#frame=pd.DataFrame(data,columns=["id","X","Y","Z"], index=[1,2,3,4,5,6])

gp = frame.groupby('id')

for group in gp:
    frame = pd.DataFrame(group[1])
    #df = list(group[1].rolling(2).mean())
    #df = list(frame.rolling(2)['X'].apply(func, args=(np.array(frame['Y'])[0],)))
    df  = list(frame.set_index('date').rolling(2)['date2'].apply(timeDiff, args=(freq, tu,)))
    df2 = list(frame.set_index('date').rolling(10)['signal'].apply(spike1, args=(mu, QA1,)))
    df3 = list(frame.set_index('date').rolling(10)['signal'].apply(spike2, args=(QA2,)))
    df4 = list(frame.set_index('date').rolling(10)['signal'].apply(spike3, args=(QA3,)))
    df5 = list(frame.set_index('date').rolling(10)['signal'].apply(spike3_mod, args=(3.5,)))
    df6 = list(frame.set_index('date').rolling(10)['signal'].apply(lowvar, args=(p_delta,)))
    df7 = list(frame.set_index('date').rolling(6)['signal'].apply(persist))
    df8 = list(frame.set_index('date').rolling(10)['signal'].apply(mzs_test, args=(3.5,)))
    df9 = list(frame.set_index('date').rolling('10H')['signal'].apply(winCount))
    df10 = list(frame.set_index('date').rolling('10H')['signal'].apply(spike4, args=(QA4, lowValue4)))
    
    #df = pd.DataFrame(df)
    df1 = np.array(group)[1]
    print(df1)
    print(" ")
    df1['QA_valid']    = df
    df1['QA_spk1']     = df2
    df1['QA_spk2']     = df3
    df1['QA_spk3']     = df4
    df1['QA_spk3_mod'] = df5
    df1['QA_LV']       = df6
    df1['QA_per']      = df7
    df1['mzs_test']    = df8
    df1['count']       = df9
    df1['QA_spk4']     = df10
#   proc = df.rolling(2).apply(func, args=(2,)).reset_index()
    df_list.append(df1)
#    print(df_list)

# conversion to JSON object 
# this will be replace by 'putData()' method exported by the Data Handler in production
df_result = pd.concat(df_list)
print(df_result)
result_json = df_result.to_json(orient='records', date_format='iso')

# visualization segment for testing (will be removed for production)
"""
fig = pylab.figure()
ax = fig.add_subplot(111)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax2.set_xlabel('time step')
ax2.set_ylabel('persist QC flag value')
ax1.set_ylabel('concentratioin (ug/cm^3)')
ax1.set_title('test time series')
pylab.subplot(211)
x=[x for x in range(100)]
color = np.where(df_result['QA_per'].iloc[13100:13200] == 1, 'red', 'skyblue')
y = df_result['signal'].iloc[13100:13200]
#y = df_result['mzs_test'].iloc[13100:13200]
ax1.vlines(x, ymin=0, ymax=y, color='skyblue', lw=2, alpha=0.8)
ax1.scatter(x, y, color=color,s=9)

ax2.set_title('Outlier Signal')

ax2.scatter(x, df_result['QA_per'].iloc[13100:13200], color="red", s=9)
ax2.vlines(x, ymin=0, ymax=df_result['QA_per'].iloc[13100:13200], color="red", lw=2)
#ax2.ylim(-1.5, 1.5)
"""
"""
fig = pylab.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax3.set_xlabel('time step')
ax3.set_ylabel('persist QC flag value')
ax1.set_ylabel('concentratioin (ug/cm^3)')
ax2.set_ylabel('modified z-score')
ax1.set_title('test time series')
#pylab.subplot(211)
x=[x for x in range(100)]
color = np.where(df_result['QA_spk2'].iloc[13100:13200] == 1, 'red', 'skyblue')
y1 = df_result['signal'].iloc[13100:13200]
y2 = df_result['mzs_test'].iloc[13100:13200]
ax1.vlines(x, ymin=0, ymax=y, color='skyblue', lw=2, alpha=0.8)
ax1.scatter(x, y, color=color,s=9)

ax2.vlines(x, ymin=0, ymax=y2, color='skyblue', lw=2, alpha=0.8)
ax2.scatter(x, y2, color=color,s=9)

ax3.scatter(x, df_result['QA_spk2'].iloc[13100:13200], color="red", s=9)
ax3.vlines(x, ymin=0, ymax=df_result['QA_spk2'].iloc[13100:13200], color="red", lw=2)
#ax2.ylim(-1.5, 1.5)
"""







    
    






