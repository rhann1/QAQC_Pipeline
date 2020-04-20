# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:23:44 2020

"""

def myfunc(x,a):
    return x[0]+4
import pandas as pd
import numpy as np

def myFunc2(x, freq):
    #print(x['B'])
    comp = int(freq*24*0.75)
    if x['overall'].loc[x['overall'] == 1].count() > comp:
        return 1
    else:
        return 0
    
def myFunc3(x):
    cnt = x['overall'].loc[x['overall'] == 1].count()
    return cnt

def validAvg(x, freq):
    x = x.loc[x['overall'] == 1]
    comp = int(freq*24*0.75)
    if len(x) > comp:
        return 1
    else:
        return 0
    
def average(x, freq): 
    print(len(x))
    print(x['A'].mean())
    y = x.loc[x['overall'] == 1]
    print(len(y))
    print(y.reset_index()['A'].mean())
    avg = y.reset_index()['A'].mean()
    print(avg)
    return avg

    
    
freq = 1
frame= pd.DataFrame()
tidx = pd.date_range('2000-01-01', '2000-01-04 23:00:00', freq='H')
frame['DateTime'] = tidx
frame['A'] = np.random.randint(1, 4, frame.shape[0])
frame['B'] = np.random.randint(0, 2, frame.shape[0])
frame['C'] = np.random.randint(0, 2, frame.shape[0])

frame['overall'] = frame['B'] | frame['C']
 
df12 = frame.rolling(1).apply(lambda x: myfunc(x,1))
dfx = frame.set_index('DateTime').groupby(pd.Grouper(freq = '1D')).apply(myFunc2, freq=1).reset_index()
dfx2 = pd.DataFrame(frame.set_index('DateTime').groupby(pd.Grouper(freq = '1D')).apply(myFunc3))
dfx2['valid'] = pd.DataFrame(frame.set_index('DateTime').groupby(pd.Grouper(freq = '1D')).apply(validAvg, freq=1))
dfx2 = dfx2.reset_index()
dfx2['D'] = dfx[0] 
 
dfx2['mean']=pd.DataFrame(frame.set_index('DateTime').resample('1D')['A'].apply(lambda x: x.mean())).reset_index()['A']
print(dfx2)
dfx2['mean']=pd.DataFrame(frame.set_index('DateTime').resample('1D').apply(lambda x: x.loc[x['overall'] == 1].mean())).reset_index()['A']
print(dfx2)
dfx2['mean']=pd.DataFrame(frame.set_index('DateTime').resample('1D').apply(average, freq=1)).reset_index()[0]
print(dfx2)
#dfx2['mean'] = pd.DataFrame(frame.set_index('DateTime').groupby(pd.Grouper(freq = '1D')).apply(average, freq=freq)).reset_index()[0]
#print(dfx2)





