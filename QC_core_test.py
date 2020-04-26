#-*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import datetime as dt
import json
from pandas.io.json import json_normalize, to_json
import pylab
import matplotlib.pyplot as plt
from TestDataSourceClass import TestDataSource


# upper instrument detection limit check (UDL)
# simple threshold comparison against the current record x[0]
def udlcheck(x, udl):
    if x[0] > udl:
        return 1
    else:
        return 0

# lower instrument detection limit (LDL)    
def ldlcheck(x, ldl):
    x = x[::-1]
    if x[0] < ldl:
        return 1
    else:
        return 0

# Simple spiike detector.
# compares value with previous record and returns potential outlier if difference is > than %delta 
# only value above a given threshold are considered
# this method is not preferable 
def spike(x, mu, delta):
    df=x
    x_bar=x.mean()                       # store the mean of the windowed values
    if np.abs(df[0] - df[1]) >= delta:   # check if the difference between current and previous is > than threshold
        return 1
    else:
        return 0
    
# mean absoute deviation based spike detection.    
def spike1(x, QA1):
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
def spike2(x, QA2):
    x = x[::-1]
    med_x = np.median(x) # find the median of the window
    mad = np.median(np.abs(np.subtract(x, -med_x))) # compute the median absolute deviation
    mzs = 0.675*(x[0] - med_x)/mad # compute the modified z-score (mzs)
    if mzs > QA2:
#        print(x, med_x, mad, mzs)
        return 1
    else: 
        return 0
   
# normalized deviation above the mean
def spike3(x, QA3):
    x = x[::-1]
    x_bar = x.mean()
    dev = np.abs(x[0]-x_bar)/x_bar
    if dev > QA3 and x_bar > lowValue3:
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

def spike3_mod(x, zs_t):
    x = x[::-1]
    x_bar = x[0:len(x)-2].mean()
    dev = np.abs(x[0]-x_bar)/x_bar
    if dev > 5 and x_bar > 12:
        print(x, x_bar, dev)
        return 1
    else:    
        return 0

# test to see if data within window varies below a given threshold value
# this still needs work 
def lowvar(x, p_delta):
    x = x[::-1]
    x_bar = x.mean()
    sdev = np.std(x)
    print(np.abs(np.subtract(x,x_bar)).sum()/len(x))
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

################################################################################################################
# function for supporting data completeness and aggregation operations on sub-hourly data
################################################################################################################
    
# function for determining if computed average is valid based on completeness requirement
# QC flagged records and missing records (gaps) do not count towards completeness
def validAvg(x, freq):
    x = x.loc[x['QA_overall'] == 0]
    comp = int(60/freq*0.75)
    if len(x) > comp:
        return 1
    else:
        return 0
    
# determines the count of valid records within the aggregation interval (example: for duration= 5min --> valud count should be 12)
# this is a utility function used for debugging purposes
def intervalCount(x, freq):
    x = x.loc[x['QA_overall'] == 0]
    comp = int(freq*12*0.75)
    if len(x) > comp:
        return len(x)
    else:
        return len(x)

# determines the percentage associtated with the completeness criteria for an averaging interval
def PercentageCompletion(x, freq, expectedCount):
    x = x.loc[x['QA_overall'] == 0]
    validRecords = len(x)
    percentageCompletion = validRecords/expectedCount*100
    return percentageCompletion

# computes the simple specified time-base average from sub-base records    
def average(x, freq): 
    print(len(x))
    print(x['AObs'].mean())
    y = x.loc[x['QA_overall'] == 0]
    print(len(y))
    print(y.reset_index()[['AObs']].mean())
    avg = y.reset_index()['AObs'].mean()
    print(avg)
    return avg


# these are supporting functions for determining if the expected number of records are occuring withing a time interval           
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

def winCount(x):
    x = x[::-1]
    return len(x)

##############################################################3 end QA function definitions    
#statiscal parameter definitions
# temporatily substitues for values that will be retrieved from the QA metadata table in the database
udl = 500
ldl = 0
mu = 15
sigma = 7
freq = 60
tu = 'h'
window = 2
delta = 20
delta1 = 10
p_delta = 0.55
expectedCount = 12

# get simulated data from data source object
tds = TestDataSource()
frame = tds.getSimulatedDataSeries()
store = frame
# the production script will use the method 'dh.getHourlyData()' and 'dh.getSubhourlyData()' from the DataHandler() class

# convert successive datetime objects to timedeltas on a unit seconds basis
#data = data.sort_values(['id','date'], ascending=False)
frame['date1'] = pd.to_timedelta(frame['StartDateTime']).astype('timedelta64[m]').astype(int)
#prepare frame for group operation
frame['date2'] = t_delta(frame['StartDateTime'])


QA1 = 3
QA2 = 4.0
QA3 = 2.1
QA4 = 3.0
lowValue3 = 5
lowValue4 = 5



# determine subhourly observations within the aggregation target time unit (standard: 1 hour for subhourly data)
# hourcount will give a count of records present for a given target hour to be used for computing the completeness requirement
# this resampling operation will support hourly aggregation of sub-hourly data
hourCount = frame.groupby('StreamId').resample('D', on='StartDateTime').count()

########################################################################################################################################################
# QC function drivers. Each driver applies the specified function operating over a moving window.  The window sizes and function parameters are sourced 
# from the QA config dataframe.  A 'groupby' is performed on the measurement data by 'StreamId' and the operations are performed on each group.

# create 'groupby' object 
gp = frame.groupby('StreamId')

# create empty sub-hourly and aggregation list object to be populated with records from the processed groups
df_list  = []
adf_list = []

# process each StreamId group
for group in gp:
    frame = pd.DataFrame(group[1])
    df   = list(frame.set_index('StartDateTime').rolling(2)[ 'date2'].apply(timeDiff, args=(freq, tu,)))
    df2  = list(frame.set_index('StartDateTime').rolling(2)[ 'AObs'].apply(spike1, args=(QA1,)))
    df3  = list(frame.set_index('StartDateTime').rolling('1H')['AObs'].apply(spike2, args=(QA2,)))
    df4  = list(frame.set_index('StartDateTime').rolling('1H')['AObs'].apply(spike3, args=(QA3,)))
    df5  = list(frame.set_index('StartDateTime').rolling('1H')['AObs'].apply(spike3_mod, args=(3.5,)))
    df6  = list(frame.set_index('StartDateTime').rolling('1H')['AObs'].apply(lowvar, args=(p_delta,)))
    df7  = list(frame.set_index('StartDateTime').rolling(6)[ 'AObs'].apply(persist))
    df9  = list(frame.set_index('StartDateTime').rolling('1H')['AObs'].apply(udlcheck, args=(udl,)))
    df10 = list(frame.set_index('StartDateTime').rolling('1H')['AObs'].apply(ldlcheck, args=(ldl,)))
    df11 = list(frame.set_index('StartDateTime').rolling('1H')[ 'AObs'].apply(spike4, args=(QA4, lowValue4)))
    df12 = list(frame.set_index('StartDateTime').rolling('1H')['AObs'].apply(winCount))
    
    # set staging frame for group
    df1 = np.array(group)[1]

    # set value for each QC flag
    df1['QA_valid']    = df
    df1['QA_spk1']     = df2
    df1['QA_spk2']     = df3
    df1['QA_spk3']     = df4
    df1['QA_spk3_mod'] = df5
    df1['QA_LV']       = df6
    df1['QA_per']      = df7
    df1['QA_udl']      = df9
    df1['QA_ldl']      = df10
    df1['QA_spk4']     = df11
    df1['winCount']    = df12

    # compute overall QC flag using bitwise logical 'or' combination of level 1 flags
    df1['QA_overall']  = df1['QA_spk3'].loc[df1['QA_spk3'].notnull()].apply(lambda x: int(x)) | \
                         df1['QA_per'].loc[df1['QA_per'].notnull()].apply(lambda x: int(x))   | \
                         df1['QA_udl'].loc[df1['QA_udl'].notnull()].apply(lambda x: int(x))   | \
                         df1['QA_ldl'].loc[df1['QA_ldl'].notnull()].apply(lambda x: int(x))   
                         
    df_list.append(df1)  # add resulting QC flags to the temporary list object for each StreamId group

    # Begin aggregation operations       
    adf1 = pd.DataFrame(frame.set_index('StartDateTime').groupby(pd.Grouper(freq = '1H')).apply(validAvg, freq=5)).rename({0:'validAvg'}, axis=1)
    adf1['StreamId'] = group[0]
    adf1['validCount'] = pd.DataFrame(frame.set_index('StartDateTime').groupby(pd.Grouper(freq = '1H')).apply(intervalCount, freq=5))
    adf1['percentCompetion'] = pd.DataFrame(frame.set_index('StartDateTime').groupby(pd.Grouper(freq = '1H')).apply(PercentageCompletion, freq=5, expectedCount = expectedCount))
    adf1['average'] = pd.DataFrame(frame.set_index('StartDateTime').resample('1H').apply(average, freq=5))[0] 
    
    adf_list.append(adf1) # add resulting averages and supporting columns to  result aggregation dataframe
    
# concatentate QC flag and aggregations list to a dataframe
df_result  = pd.concat(df_list)
adf_result = pd.concat(adf_list)

# write testing results to output files
df_result.to_csv('testing_results/test_result.csv')
adf_result.to_csv('testing_results/test_avgs.csv')
print(df_result)

# end function drivers
####################################################################################################################################################

# conversion to JSON object 
# this will be replace by 'putData()' method exported by the Data Handler in production


# dh.putData() # this call will transfer QC flag dataframes back to the data handler

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
y = df_result['AObs'].iloc[13100:13200]
#y = df_result['mzs_test'].iloc[13100:13200]
ax1.vlines(x, ymin=0, ymax=y, color='skyblue', lw=2, alpha=0.8)
ax1.scatter(x, y, color=color,s=9)

ax2.set_title('Outlier AObs')

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
y1 = df_result['AObs'].iloc[13100:13200]
y2 = df_result['mzs_test'].iloc[13100:13200]
ax1.vlines(x, ymin=0, ymax=y1, color='skyblue', lw=2, alpha=0.8)
ax1.scatter(x, y1, color=color,s=9)

ax2.vlines(x, ymin=0, ymax=y2, color='skyblue', lw=2, alpha=0.8)
ax2.scatter(x, y2, color=color,s=9)

ax3.scatter(x, df_result['QA_spk2'].iloc[13100:13200], color="red", s=9)
ax3.vlines(x, ymin=0, ymax=df_result['QA_spk2'].iloc[13100:13200], color="red", lw=2)
#ax2.ylim(-1.5, 1.5)
"""







    
    






