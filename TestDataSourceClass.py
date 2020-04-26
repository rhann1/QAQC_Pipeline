# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:49:26 2020

@author: raifo
"""

import pandas as pd
import numpy as np

class TestDataSource:
    
    def __init__(self, mu = 15.0, sigma = 7):
        self.mu = mu
        self.sigma = sigma
        

    def getSimulatedDataSeries(self):
        
        #tidx = pd.date_range('2000-01-01', '2000-12-31 23:00', freq='H') # StartDateTime index for hourly simulation
        tidx = pd.date_range('2000-01-01 00:00:00', '2000-01-01 23:00', freq='5min') # StartDateTime for sub-hourly simulation
        width = len(tidx)
    
        s = np.random.normal(self.mu, self.sigma, width)
        s[int(width/2)] = 90
        s1 = np.random.normal(self.mu, self.sigma, width)
        s1[int(width/2)+100] = 275
        s1[int(width/4):int(width/4)+20] = 12.4
        #generation of a simulated DateTime sequence
        #tidx = pd.date_range('2000-01-01', '2000-12-31 23:00', freq='H') # StartDateTime index for hourly simulation
        #tidx = pd.date_range('2000-01-01 00:00:00', '2000-01-01 23:00', freq='5min') # StartDateTime for sub-hourly simulation
    
        #creation of time series dataframe
        frame = pd.DataFrame({'StartDateTime':tidx,'AObs':s})
        frame2 = pd.DataFrame({'StartDateTime':tidx,'AObs':s1})
        #frame2['StartDateTime'].iloc[8780] = pd.to_datetime('2000-12-31 20:30:00')    #test row for timediff check
        #frame2.drop(index=8774, inplace=True)

        # initialize two group IDs
        frame['StreamId']=0
        frame2['StreamId']=1
        # append ID=1 group
        frame = frame.append(frame2, ignore_index=True)

        #data = pd.read_csv('sample_hourly_data/OCAP_SJV.csv')
        frame = pd.read_csv('sample_subhourly_data/2020-02-08_1.csv')
        #data.rename(columns={'DateTime':'StartDateTime', 'MeasuredValue':'AObs', 'SiteName':'StreamId'}, inplace=True)
        frame.rename(columns={'MonitorId':'StreamId'}, inplace=True)
        frame['StartDateTime'] = pd.to_datetime(frame['StartDateTime'])    
        frame = frame.loc[frame['Parameter'] == 88101]
        return frame

if __name__ == "__main__":
    
    # create data source object
    tds = TestDataSource()
    data = tds.getSimulatedDataSeries()
    print(data)
    
    
    