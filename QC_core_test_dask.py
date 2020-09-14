#-*- coding: utf-8 -*-

import dask as dd
from dask.distributed import Client
from dask import dataframe

import numpy as np
import datetime as dt
import json
import pandas as pd 
from pandas.io.json import json_normalize, to_json
import pylab
import matplotlib.pyplot as plt
from TestDataSourceClass import TestDataSource
from DataHandler import DataHandler
import sys
import time
import swifter
import multiprocessing as mp


def QC_Core(IsSubHourly, measurementFrame, configFrame):

    # upper instrument detection limit check (UDL)
    # simple threshold comparison against the current record x[0]
    def udlcheck(x, udl, useUDL):
        if pd.isnull(udl): # check for NULL valued metadata
            return -1
        if useUDL < 1:
            return -3
        x = x[::-1]
        if x[0] > udl:
            return 1
        else:
            return 0
    
    # lower instrument detection limit (LDL)    
    def ldlcheck(x, ldl, useLDL):
        if pd.isnull(ldl):
            return -1
        if useLDL < 1:
            return -3
        x = x[::-1]
        if x[0] < ldl:
            return 1
        else:
            return 0
    
    # median deviation based spike detection.    
    def spike1(x, QA1, useQC1):
        if pd.isnull(QA1):
            return -1
        if useQC1 < 1:
            return -3
        x = x[::-1]
        x_med = np.median(x)
        dev = np.abs(x[0] - x_med)/x_med
        if dev > QA1 and x_med > 0:
            return 1
        else:
            return 0
    
    # modified z-score spike detector.  This is a robust method based on the median absolute deviation.
    # at this point this is the preferable method for detection obvious outliers
    def spike2(x, QA2, useQC2):
        x = np.array(x)
        #print('threshold = ' + str(QA2))
        if pd.isnull(QA2):
            return -1
        if useQC2 < 1:
            return -3
        x = x[::-1]
        b = [0, 0, 0, 0, 1.363, 1.206, 1.200, 1.140, 1.129, 1.107] # quartile correction factors for window widths < 10
        #b = [0, 0, 0, 0, 1.500, 1.363, 1.206, 1.200, 1.140, 1.129, 1.107]
        k = 1.4826 # reciprocal of quartile function for a normal distribution
        q = QA2 # MAD threshold value from QC metadata table. The larger the value of 'q', the lower the probability of capturing an outlier.  Use q=10 for very large deviations.
        
        if len(x) < 4: return -1
        # compute distribution quartile correction factor based on number of windowed observations
        if len(x) < 10:
            b_n = b[len(x)]
        else:
            b_n = len(x)/(len(x) - 0.8)
            
        med_x = np.median(x) # find the median of the window
        mad_adj = np.median(np.abs(np.subtract(x, -med_x)))*b_n*q*k # compute the adjusted median absolute deviation
        #print(mad_adj, med_x, len(x))     
        if ((med_x - mad_adj >= x[0]) or (med_x + mad_adj <= x[0])) and mad_adj > 0:
            return 1
        else: 
            return 0
        
    # this function distinguishes a true isoloated spike [ 0 0 100 0 0 ] from a spike-ramp [0 0 100 80 40]
    def true_spike(x, QA2):
        x = x[::-1]  
        x_0 = int(len(x)/2)
        xLeft = x[:x_0+1]
        xLeft = xLeft[::-1]
        xRight = x[x_0:]
        qLeft = spike2(xLeft, QA2, useQC2)
        qRight = spike2(xRight, QA2, useQC2)
        #print(np.array(x), xLeft, xRight, np.median(xLeft), np.median(xRight), len(xLeft), len(xRight))
        if qLeft == 1 and qRight == 1:
            return 1
        else:
            return 0
            
    # standard score (z-score)
    def spike3(x, QA3, useQC3):
        if pd.isnull(QA3):
            return -1
        if useQC3 < 1:
            return -3
        x = x[::-1]
        x_bar = x.mean()
        x_sdev = np.std(x)
        z_score = np.abs((x[0] - x_bar)/x_sdev)
        if z_score > QA3 and np.abs(x_sdev) > 0 and x[1] > 0.4:
            return 1
        else:
            return 0
    
    # IQR outlier detection.
    # This method needs a large observation window and may not be feasible as a Level 1 QC test
    def spike4(x, QA4):
        if pd.isnull(QA4):
            return -1
        x = x[::-1]
        q25, q75 = np.percentile(x, 25), np.percentile(x, 75)
        iqr = q75 - q25
        threshold = iqr*QA4
        lower, upper = q25 - threshold, q75 + threshold
        if x[0] > upper:
            return 1
        else: 
            return 0
    
    # test to see if data within window varies below a given threshold value
    # this still needs work (not implemented to compute flag yet)
    def lowvar(x, p_delta):
        if pd.isnull(p_delta):
            return -1
        x = x[::-1]
        x_bar = x.mean()
        sdev = np.std(x)
        if np.abs(np.subtract(x,x_bar)).sum()/len(x) < p_delta*sdev:
            return 1
        else:
            return 0
    
    # persistent value test.  This will check if the current observation has repeated over the length of the test window within a specified varience.    
    def persist(x, persistThresh, usePersistCount):
        if usePersistCount < 1:
            return -3
        x = x[::-1]
        if x[0] < persistThresh:
            return 0
        sum_dev = sum([x[0]-x[i] for i in range(len(x))]) # computes the sum of the magnitudes of the deviation from x[0]. If the sum ==0, x[0] is a repeating value in the series. 
        if sum_dev == 0: # Check if observation is part of a repeating sequence.
            return 1
        else:
            return 0

    # checks for identical values in the test window    
    def persist2(x, persistThresh, usePersistCount):
        if usePersistCount < 1:
            return -3
        x = x[::-1]
        if x[0] < persistThresh:
            return 0
        if len(set(x)) <= 1: # Check if observation is part of a repeating sequence.
            return 1
        else:
            return 0
        
    def modZScore(x):
        x = x[::-1]
        med_x = np.median(x) # find the median of the window
        mad = np.median(np.abs(np.subtract(x, -med_x))) # compute the median absolute deviation
        mzs = 0.675*(x[0] - med_x)/mad # compute the modified z-score (mzs)
        return mzs
    
    ################################################################################################################
    # function for supporting data completeness and aggregation operations on sub-hourly data
    ################################################################################################################
        
    # function for determining if computed average is valid based on completeness requirement
    # QC flagged records and missing records (gaps) do not count towards completeness
    def validAvg(x, freq):
        if pd.isnull(freq):
            return -1
        x = x.loc[x == 0]
        comp = int(60/freq*0.75)
        if len(x) >= comp:
            return 1
        else:
            return 0
        
    # determines the count of valid records within the aggregation interval (example: for duration= 5min --> valud count should be 12)
    # this is a utility function used for debugging purposes
    def intervalCount(x, freq):
        x = x.loc[x == 0]
        comp = int(freq*12*0.75)
        if len(x) > comp:
            return len(x)
        else:
            return len(x)
    
    # determines the percentage associtated with the completeness criteria for an averaging interval
    def PercentageCompletion(x, freq, expectedCount):
        x = x.loc[x == 0]
        validRecords = len(x)
        percentageCompletion = validRecords/expectedCount*100
        return percentageCompletion
    
    # computes the simple specified time-base average from sub-base records    
    def average(x, freq): 
        y = x.loc[x == 0]
        avg = y.mean()
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
    
    # supporting function to check if the rolling window meets a 75% completeness requirment
    # Window maintains a fixed width.  Missing observations are counted as temporal gaps and can not exceed the completeness threshold
    def spikeValid(x, freq, width):
        x = x[::-1]
        l = len(x)
        t_expected = freq*(width - 1)
        t_delta = x[0] - x[l-1] # compute time delta associated with the window 
        gaps = (t_delta - t_expected)/freq
        gap_fraction = (width - gaps)/width
        if gap_fraction < 0.75: # if time delta in larger than expected based on freq, mark window as invalid (missing observations in time interval)
            return 1
        else:
            return 0
    
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
    expectedCount = 4
    
    # get simulated data from data source object
    tds = TestDataSource()
    # retrieve measurement dataframe from simulated source
    frame = tds.getSimulatedDataSeries()
    # retrieve measurement dataframe sourced from API endpoint
    frame = measurementFrame
    print(frame)
    
    #frame.set_value(3, 'AObs', np.nan)
    # convert StartDateTime column to datetime objects
    frame['StartDateTime'] = pd.to_datetime(frame['StartDateTime'])
    # sort API sourced frame in DESC order    
    frame = frame.sort_values(['StreamSegmentId', 'StartDateTime'], ascending=True)
    
    #############################################################################################################
    ## TO BE REMOVED ##
    # temporary simulated time-series for testing centered window
    """
    StartDateTime = pd.date_range('2000-1-1 00:00:00', periods=72, freq='5min')
    simFrame = pd.DataFrame(data=np.random.randn(72)+30, index=StartDateTime, columns=['AObs'])
    frame = simFrame.reset_index()
    frame['StreamSegmentId'] = 1111
    frame['Parameter'] = '99999'
    frame.columns = ['StartDateTime', 'AObs', 'StreamSegmentId', 'Parameter']
    frame.loc[24, 'AObs'] = 1000
    frame.loc[25, 'AObs'] = 1000
    frame.loc[26, 'AObs'] = 1000
    frame.loc[27, 'AObs'] = 1000
    
    frame.loc[37, 'AObs'] = 1000
    
    
    frame.loc[40, 'AObs'] = 1000
    frame.loc[41, 'AObs'] = 1000
    frame.loc[42, 'AObs'] = 120
    frame.loc[43, 'AObs'] = 120
    
    frame.drop(frame.index[40], inplace=True)
    frame.drop(frame.index[41], inplace=True)
    """
    
    ## TO BE REMOVED ##
    #############################################################################################################
    
    # get QA metadata from local CSV file (would normally be sourced from API call)
    configData = tds.getQAConfigDataFromFile('sample_QA_metadata/QC_SM_OneYear.csv')
    configData = configFrame
    
    # get QA metadata sourced from API endpoint
    #configData = configFrame

    configData.replace(r'^\s*$', np.nan, regex=True, inplace=True) # replace any white space with NaN. Unwanted whitespace instead of NULL is sometimes generated by Excel
    store = frame
    #frame = pd.merge(frame, configData, how='inner', on='StreamSegmentId')
    
    # the production script will use the method 'dh.getHourlyData()' and 'dh.getSubhourlyData()' from the DataHandler() class
    
    # convert successive datetime objects to timedeltas on a unit seconds basis
    #data = data.sort_values(['id','date'], ascending=False)
    #frame['date1'] = pd.to_timedelta(frame['StartDateTime']).astype('timedelta64[m]').astype(int)
    #prepare frame for group operationdf
    frame['date1'] = t_delta(frame['StartDateTime']).astype(int)
    
    
    QA1 = 2.0
    QA2 = 4
    QA3 = 4
    QA4 = 4
    lowValue3 = 0
    lowValue4 = 0
    
    
    
    # determine subhourly observations within the aggregation target time unit (standard: 1 hour for subhourly data)
    # hourcount will give a count of records present for a given target hour to be used for computing the completeness requirement
    # this resampling operation will support hourly aggregation of sub-hourly data
    hourCount = frame.groupby('StreamSegmentId').resample('D', on='StartDateTime').count()
    
    ########################################################################################################################################################
    # QC function drivers. Each driver applies the specified function operating over a moving window.  The window sizes and function parameters are sourced 
    # from the QA config dataframe.  A 'groupby' is performed on the measurement data by 'StreamSegmentId' and the operations are performed on each group.
    
    # create 'groupby' object 
    gp = frame.groupby('StreamSegmentId')
    
    # create empty sub-hourly and aggregation list object to be populated with records from the processed groups
    df_list  = []
    adf_list = []
    #print(frame)
    
    # create dask scheduler
    client = Client(n_workers=mp.cpu_count()-6, threads_per_worker=2) # select 'in proc' schedule    
    # process each StreamSegmentId group
    for group in gp:
        
        # process only if QC metadata exists
        if len(configData.loc[configData['StreamSegmentId'] == group[0]].values) != 0:
            print('Processing StreamSegmentId = ' + str(group[0]))
        
            udl                = float(configData.loc[configData['StreamSegmentId'] == group[0]]['UDL'].values[0])
            ldl                = float(configData.loc[configData['StreamSegmentId'] == group[0]]['LDL'].values[0])
            mdl                = float(configData.loc[configData['StreamSegmentId'] == group[0]]['MDL'].values[0])
            durationMinutes    = float(configData.loc[configData['StreamSegmentId'] == group[0]]['SamplingDurationMinutes'].values[0])
            persistCount       = int(configData.loc[configData['StreamSegmentId'] == group[0]]['PersistCount'].values[0])
            persistThresh      = float(configData.loc[configData['StreamSegmentId'] == group[0]]['PersistThresh'].values[0])
            QC1                = float(configData.loc[configData['StreamSegmentId'] == group[0]]['Anom1Thresh'].values[0])
            QC2                = float(configData.loc[configData['StreamSegmentId'] == group[0]]['Anom2Thresh'].values[0])
            QC3                = float(configData.loc[configData['StreamSegmentId'] == group[0]]['Anom3Thresh'].values[0])
            QC4                = float(configData.loc[configData['StreamSegmentId'] == group[0]]['Anom4Thresh'].values[0])
            useUDL             = int(configData.loc[configData['StreamSegmentId'] == group[0]]['useUDL'].values[0])
            useLDL             = int(configData.loc[configData['StreamSegmentId'] == group[0]]['useLDL'].values[0])
            useMDL             = int(configData.loc[configData['StreamSegmentId'] == group[0]]['useMDL'].values[0])
            usePersistCount    = int(configData.loc[configData['StreamSegmentId'] == group[0]]['usePCount'].values[0])
            useQC1             = int(configData.loc[configData['StreamSegmentId'] == group[0]]['useA1Check'].values[0])
            useQC2             = int(configData.loc[configData['StreamSegmentId'] == group[0]]['useA2Check'].values[0])
            useQC3             = int(configData.loc[configData['StreamSegmentId'] == group[0]]['useA3Check'].values[0])
            useQC4             = int(configData.loc[configData['StreamSegmentId'] == group[0]]['useA4Check'].values[0])
            useQC7             = int(configData.loc[configData['StreamSegmentId'] == group[0]]['useA7Check'].values[0])
            QC1WinSize         = int(configData.loc[configData['StreamSegmentId'] == group[0]]['Anom1WindowSize'].values[0])
            QC2WinSize         = int(configData.loc[configData['StreamSegmentId'] == group[0]]['Anom2WindowSize'].values[0])
            QC3WinSize         = int(configData.loc[configData['StreamSegmentId'] == group[0]]['Anom3WindowSize'].values[0])
            QC4WinSize         = int(configData.loc[configData['StreamSegmentId'] == group[0]]['Anom4WindowSize'].values[0])
            persistWinSize     = int(configData.loc[configData['StreamSegmentId'] == group[0]]['PersistWindowSize'].values[0])
            computeQuality     = int(configData.loc[configData['StreamSegmentId'] == group[0]]['computeQuality'].values[0])
            #useAdj             = int(configData.loc[configData['StreamSegmentId'] == group[0]]['useAdj'].values[0])
            
            if computeQuality < 1:
                useQC1 = 0;
                useQC2 = 0;
                useQC3 = 0
                useQC7 = 0
                usePersistCount = 0
                useUDL = 0
                useLDL = 0
                useMDL = 0
                
            
            
        
    
            frame = pd.DataFrame(group[1])
            frame = frame.set_index('StartDateTime')
            
            # create dask dataframe
            ddf = dd.dataframe.from_pandas(frame, npartitions=mp.cpu_count()-6)
            
            """if useAdj == 1:
                frame['AObs'] == frame['AObsAdj']
            """
            
            
            # compute lists of QC flags for each test
            #df    = list(frame.set_index('StartDateTime').rolling(2)['date2'].apply(timeDiff, args=(freq, tu,)))
            df2   = list(ddf.AObs.rolling(QC1WinSize).apply(spike1, args=(QC1, useQC1), raw=True).compute())
            #df3   = list(frame.set_index('StartDateTime').rolling(QC2WinSize)['AObs'].apply(spike2, args=(QA2,)).shift(-30, freq='m'))
            df3   = list(ddf.AObs.rolling(QC2WinSize).apply(spike2, args=(QC2, useQC2,), raw=True).compute())
            df4   = list(ddf.AObs.rolling(QC3WinSize).apply(spike3, args=(QC3, useQC3,), raw=True).compute())
            df5   = list(ddf.AObs.rolling(QC3WinSize).apply(spike2, args=(3.5, useQC2,), raw=True).shift(-15, freq='m').compute())
            df5b  = list(ddf.AObs.rolling(QC3WinSize, min_periods=1).apply(true_spike, args=(3.2,), raw=True).shift(-30, freq='m').compute())
            df6   = list(ddf.AObs.rolling(persistWinSize).apply(lowvar, args=(p_delta,), raw=True))
            df7   = list(ddf.AObs.rolling(persistWinSize).apply(persist, args=(persistThresh, usePersistCount,), raw=True).compute())
            df8   = list(ddf.AObs.rolling(persistWinSize).apply(persist2, args=(persistThresh, usePersistCount,), raw=True).compute())
            df9   = list(ddf.AObs.rolling(2).apply(udlcheck, args=(udl, useUDL,), raw=True).compute())
            df10  = list(ddf.AObs.rolling(2).apply(ldlcheck, args=(ldl, useLDL,), raw=True).compute())
            df11  = list(ddf.AObs.rolling(QC4WinSize).apply(spike4, args=(QC4,), raw=True).compute())
            #df12  = list(frame.set_index('StartDateTime').rolling('1H')['AObs'].apply(winCount))
            #df13  = list(frame.set_index('StartDateTime')['AObs'].swifter.rolling(QC2WinSize).apply(modZScore, raw=True))
            
                      
            df2 = pd.Series(df2).fillna(-3)
            df3 = pd.Series(df3).fillna(-3)
            df4 = pd.Series(df4).fillna(-3)
            df7 = pd.Series(df7).fillna(-3)
            df8 = pd.Series(df8).fillna(-3)
            #df9 = pd.Series(df9).fillna(-3)
            #df10 = pd.Series(df10).fillna(-3)
            
            """
            sFrame = pd.DataFrame({'df2': df2,'df3': df3, 'df4': df4})
            sFrame['dfspike'] = 0
            sFrame['dfspike'] = sFrame.loc[(sFrame['df2'].notnull().astype(int) < 0) | (sFrame['df3'].notnull().astype(int) < 0) | (sFrame['df4'].notnull().astype(int) < 0)] = -3
            sFrame['dfspike'] = sFrame.loc[(sFrame['df2'].notnull().astype(int) == 1) & (sFrame['df3'].notnull().astype(int) == 1) & (sFrame['df4'].notnull().astype(int) == 1)] = 1
            sFrame['dfspike'] = sFrame.loc[(sFrame['df2'].notnull().astype(int) == 0) | (sFrame['df3'].notnull().astype(int) ==0) | (sFrame['df4'].notnull().astype(int) ==0)] = 0
            """
            # best logic so far for handling both pass/fail composite spike logic and status codes.  This still needs so work.
            dspike = list([1 if (int(x) == 1) & (int(y) == 1) & (int(z) == 1) else 0 for x,y,z in zip(df2,df3,df4)])
            dspike2 = list([-3  if (int(x) < 0 ) | (int(y) < 0) | (int(z) < 0) else ds for x,y,z,ds in zip(df2,df3,df4,dspike) ])
            
            #compute composite spike flag (bitwise and of QF01, QF02 and QF03)
            """dfspike = list(np.array([int(x) for x in pd.Series(df2).fillna(-3).tolist()]) & \
                           np.array([int(x) for x in pd.Series(df3).fillna(-3).tolist()]) & \
                           np.array([int(x) for x in pd.Series(df4).fillna(-3).tolist()]))
            
            pd.DataFrame({'df2': df2,'df3': df3, 'df4': df4, 'dfspike': dfspike})

            
            dfspike_status = [d if d < 0 else 0 for d in \
                             list(np.array([int(x) for x in pd.Series(df2).fillna(-3).tolist()]) | \
                                  np.array([int(x) for x in pd.Series(df3).fillna(-3).tolist()]) | \
                                  np.array([int(x) for x in pd.Series(df4).fillna(-3).tolist()]))]
            
            dfspike_ws = [d|s for d,s in zip(dfspike, dfspike_status)]
            dfspike_ws = [d for d in list(np.array(dfspike) | np.array(dfspike_status))]
            """
            
            #print(pd.DataFrame({'df2': df2,'df3': df3, 'df4': df4, 'dfspike': dfspike, 'status': dfspike_status, 'c_spike': dfspike_ws}))
            
            
            
            # these window drivers are experimental at the moment and use a time shifted offese to evaluate a target record at the mid-window point
            # this will allow distinction between a spike anomaly and ramp concentration profile (e.g., [1,1,1,100,1,1,1] vs [100,80,60,40,20,10,0]) 
            #df5   = list(frame.set_index('StartDateTime').rolling(QC3WinSize)['AObs'].apply(spike3_mod, args=(3.5,)).shift(-15, freq='m'))
            #df5b  = list(frame.set_index('StartDateTime').rolling(QC3WinSize, min_periods=1)['AObs'].apply(spike3_mod, args=(3.5,)).shift(-15, freq='m'))
            
            # determine validity of window-based QC test.  Window must be of the expected representative time interval to be a valid test.
            df2v  = list(ddf.date1.rolling(QC2WinSize).apply(spikeValid, args=(durationMinutes, QC2WinSize,),raw=True).compute())
            df3v  = list(ddf.date1.rolling(QC3WinSize).apply(spikeValid, args=(durationMinutes, QC3WinSize,),raw=True).compute())
            df4v  = list(ddf.date1.rolling(QC4WinSize).apply(spikeValid, args=(durationMinutes, QC4WinSize,),raw=True).compute())
            df5v  = list(ddf.date1.rolling(QC3WinSize).apply(spikeValid, args=(durationMinutes, QC3WinSize,),raw=True).compute())
            df6v  = list(ddf.date1.rolling(persistWinSize).apply(spikeValid, args=(durationMinutes, persistWinSize,),raw=True).compute())
            df7v  = list(ddf.date1.rolling(persistWinSize).apply(spikeValid, args=(durationMinutes, persistWinSize,),raw=True).compute())
            df8v  = list(ddf.date1.rolling(persistWinSize).apply(spikeValid, args=(durationMinutes, persistWinSize,),raw=True).compute())
            
            del ddf
            
            frame = frame.reset_index()
            
            # set value of QC flags with window validity check that are based on previous observations (e.g., outlier tests)
            # a flag value of -2 is assigned if the window-based check fails to meet the input completeness criteria
            df2 = [d if v < 1 else -2 for d,v in zip(df2,df2v)]
            df3 = [d if v < 1 else -2 for d,v in zip(df3,df3v)]
            df4 = [d if v < 1 else -2 for d,v in zip(df4,df4v)]
            df5 = [d if v < 1 else -2 for d,v in zip(df5,df5v)]
            df6 = [d if v < 1 else -2 for d,v in zip(df6,df6v)]
            df7 = [d if v < 1 else -2 for d,v in zip(df7,df7v)]
            df8 = [d if v < 1 else -2 for d,v in zip(df8,df8v)]
            
            #overall1 = list([1 if (int(w) == 1) | (int(x) == 1 ) | (int(y) == 1) | (int(z) == 1) else 0 for w,x,y,z in zip(df8,df9,df10,dspike2)])
            #overall2 = list([-3  if (int(w) < 0) | (int(x) < 0 ) | (int(y) < 0) | (int(z) < 0) else o for w,x,y,z,o in zip(df8,df9,df10,dspike2,overall1)])
            
            # set staging frame for group
            df1 = np.array(group)[1]
    
            # set value for each QC flag
            #df1['QA_valid']    = df
            df1['QF01']     = df2
            df1['QF02']     = df3
            df1['QF03']     = df4
            #df1['QA_LV']       = df6
            df1['QF04']      = df8
            #df1['repeat']    = df8
            df1['QF05']      = df9
            df1['QF06']      = df10
            df1['QF07']      = dspike2
            #df1['overall2']  = overall2
            #df1['winCount']    = df12
            #df1['mzs']         = df13
            
            #df1.replace(np.nan, -2, regex=True, inplace=True) # replace any NaN values with -2 indicating flag could not be computed due to insufficient window.
    
            # compute overall QC flag using bitwise logical 'or' combination of level 1 flags
            
            
            if computeQuality < 1:
                df1['QA_overall'] = -3
            else:
                df1['QA_overall']  = (df1['QF07'].loc[df1['QF07'].notnull()].apply(lambda x: int(x))*useQC7             | \
                                      df1['QF04'].loc[df1['QF04'].notnull()].apply(lambda x: int(x))*usePersistCount    | \
                                      df1['QF05'].loc[df1['QF05'].notnull()].apply(lambda x: int(x))*useUDL             | \
                                      df1['QF06'].loc[df1['QF06'].notnull()].apply(lambda x: int(x))*useLDL).astype(int)
                
                # this repairs issue with gap in spike flag status logic.  This needs to be replace with improved flag logic
                df1['QA_overall'].loc[df1['QF07'] < 0] = -3
                df1['QF07'].loc[(df1['QF01'] < 0) | (df1['QF02'] < 0) | (df1['QF03'] < 0)] = -3
                
            # QA_overall must meet completeness criteria (proposed 75% of input flags must be valid)
            # 'validOverall' is a list of completeness measures used for computing QA_overall'.
            #### THIS SEGMENT NEEDS REPAIR: THE COMPLETION VALUE IS COMPUTED CORRECTLY BUT THE OVERALL FLAG ALWAYS ASSERTS HIGH. ####
            validOverall = [(len(y)-sum(i<0 for i in y))/len(y) for y in [list(a) for a in zip(df2,df3,df4,df5,df6)]] # input flags used for computing QAoverall in zip().
            df1['QA_overall'] = [d if v >= 0.75 else -3 for d,v in zip(df1['QA_overall'], validOverall)]
            
                             
            df_list.append(df1)  # add resulting QC flags to the temporary list object for each StreamSegmentId group
    
            # Begin sub-hourly to hourly aggregation operations
            # only aggregate sub-hourly streams
            
            #ddf1 = dd.dataframe.from_pandas(df1.set_index('StartDateTime'), npartitions = 4)
                        
            if IsSubHourly:
                #adf1 = pd.DataFrame(df1.set_index('StartDateTime')['QA_overall'].groupby(pd.Grouper(freq = '1H')).apply(validAvg, freq=durationMinutes)).rename({0:'validAvg'}, axis=1)
                adf1 = pd.DataFrame(df1.groupby(pd.Grouper(key='StartDateTime', freq = '1H'))['AObs'].apply(validAvg, freq=durationMinutes)).rename({0:'validAvg'}, axis=1)
                adf1['StreamSegmentId'] = group[0]
                adf1['validCount'] = pd.DataFrame(df1.set_index('StartDateTime')['QA_overall'].groupby(pd.Grouper(freq = '1H')).apply(intervalCount, freq=durationMinutes))
                adf1['percentCompetion'] = pd.DataFrame(df1.set_index('StartDateTime')['QA_overall'].groupby(pd.Grouper(freq = '1H')).apply(PercentageCompletion, freq=durationMinutes, expectedCount = expectedCount))
                #adf1['average'] = pd.DataFrame(frame.set_index('StartDateTime').resample('1H').apply(average, freq=5))[0] 
                adf1['average'] = pd.DataFrame(df1.set_index('StartDateTime')['AObs'].groupby(pd.Grouper(freq = '1H')).apply(average, freq=durationMinutes))
                adf_list.append(adf1) # add resulting averages and supporting columns to  result aggregation dataframe
            
            """    
            if IsSubHourly:
                adf1 = pd.DataFrame(ddf1.QA_overall.groupby(pd.Grouper(freq = '1H')).apply(validAvg, freq=durationMinutes)).rename({0:'validAvg'}, axis=1).compute()
                adf1['StreamSegmentId'] = group[0]
                adf1['validCount'] = pd.DataFrame(df1.set_index('StartDateTime')['QA_overall'].groupby(pd.Grouper(freq = '1H')).apply(intervalCount, freq=durationMinutes))
                adf1['percentCompetion'] = pd.DataFrame(df1.set_index('StartDateTime')['QA_overall'].groupby(pd.Grouper(freq = '1H')).apply(PercentageCompletion, freq=durationMinutes, expectedCount = expectedCount))
                #adf1['average'] = pd.DataFrame(frame.set_index('StartDateTime').resample('1H').apply(average, freq=5))[0] 
                adf1['average'] = pd.DataFrame(df1.set_index('StartDateTime')['AObs'].groupby(pd.Grouper(freq = '1H')).apply(average, freq=durationMinutes))
                adf_list.append(adf1) # add resulting averages and supporting columns to  result aggregation dataframe
           """
           
    
            # concatentate QC flag and aggregations list to a dataframe
            if len(df_list) != 0:
                df_result  = pd.concat(df_list)
                # sort for local files as source
                #df_result.sort_values(['MonitorId', 'Parameter', 'StartDateTime'], ascending=False)
                # sort for API sourced measurement data
                df_result.sort_values(['StreamSegmentId', 'StartDateTime'], ascending=False)
                #df_result.to_csv('testing_results/test_result.csv')    
    
            # create hourly average dataframe
            if len(adf_list) != 0 and IsSubHourly:
                adf_result = pd.concat(adf_list)
                #adf_result.to_csv('testing_results/test_avgs.csv')
      
                #subHourlyQFlags = df_result[['MeasurementID', 'StreamSegmentId', 'StartDateTime', 'QF01', 'QF02', 'QF03', 'QF04', 'QOverall']]
                
    #print(df_result)
    df_result.to_csv('testing_results/test_result.csv')
    adf_result.to_csv('testing_results/test_avgs.csv')
    
    # end function drivers
    if IsSubHourly:
        return df_result, adf_result
    else:
        return df_result
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
    
    frame = df_result[['MeasurementID', 'StreamSegmentId', 'StartDateTime', 'QF01', 'QF02', 'QF03', 'QF04', 'QOverall']]
    """


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        IsSubHourly=True
        QaScriptId = 1
        QaProcessingLogId = 1
    else:
        QaScriptId = sys.argv[0]
        #QaProcessingLogId = sys.argv[2]
        
    intervalHoursForHourly = 1
    intervalHoursForSubHourly = 1
    maxStreamLimit = 5000
    
    #set mode of processing (testing using local files: testMode=True, processing from APIs: testMode=False)
    testMode=True
    
    #performation data exchange operations    
    #get measurement and configuration data
    if testMode:
        tds = TestDataSource()
        beginGetData = time.time()
        frame = tds.getSimulatedDataSeries()
        endGetData = time.time()
        configData = tds.getQAConfigDataFromFile('sample_QA_metadata/QC_SM_OneYear.csv')
        configData.replace(r'^\s*$', np.nan, regex=True, inplace=True) # replace any white space with NaN. Unwanted whitespace instead of NULL is sometimes generated by Excel
        start = time.time()
        computedQFlagFrame, subHourlyAggregations = QC_Core(True, frame, configData)
        end = time.time()
        print("time to retrieve data set = " + str(endGetData - beginGetData))
        print("execution time = " + str(end - start))
    else:
        dh = DataHandler()
        
        """
        # SubHourly data processing
        QaProcessingStart = time.time()
        measurementFrame, configFrame = dh.getData(False, intervalHoursForSubHourly, maxStreamLimit) # getData() returns tuple of dataframes.  Passed argument is 'IsSubHourly'.
        computedQFlagFrame, subHourlyAggregations = QC_Core(False, measurementFrame, configFrame) # QC flags and aggregations are returned
        dh.PutData(True, computedQFlagFrame) # Boolean argument is 'IsSubHourly'
        dh.PutData(False, subHourlyAggregations)
        """
        
        # Hourly data processing
        measurementFrame, configFrame = dh.getData(False, intervalHoursForHourly, maxStreamLimit) # getData() returns tuple of dataframes.  Passed argument is 'IsSubHourly'.
        #computedQFlagFrame, subHourlyAggregations = QC_Core(False, measurementFrame, configFrame) # QC flags
        processedFrames = QC_Core(False, measurementFrame, configFrame) # QC flags
        #dh.PutData(True, computedQFlagFrame) # Boolean argument is 'IsSubHourly'
        #dh.PutData(False, subHourlyAggregations)
        #QaProcessingEnd = time.time()
        
        #dh.PutProcessingLogData(QaProcessingLog) # transfer processing log data to database





    
    






