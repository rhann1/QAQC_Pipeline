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
    
        s = np.random.normal(self.mu, self.sigma, 8784)
        s[4000] = 90
        s1 = np.random.normal(self.mu, self.sigma, 8784)
        s1[4400] = 275
        s1[4370:4390] = 12.4
        #generation of a simulated DateTime sequence
        tidx = pd.date_range('2000-01-01', '2000-12-31 23:00', freq='H')
    
        #creation of time series dataframe
        frame = pd.DataFrame({'StartDateTime':tidx,'AObs':s})
        frame2 = pd.DataFrame({'StartDateTime':tidx,'AObs':s1})
        frame2['StartDateTime'].iloc[8780] = pd.to_datetime('2000-12-31 20:30:00')    #test row for timediff check
        frame2.drop(index=8774, inplace=True)

        # initialize two group IDs
        frame['StreamId']=0
        frame2['StreamId']=1
        # append ID=1 group
        frame = frame.append(frame2, ignore_index=True)

        data = pd.read_csv('sample_hourly_data/OCAP_SJV.csv')
        data.rename(columns={'DateTime':'StartDateTime', 'MeasuredValue':'AObs', 'SiteName':'StreamId'}, inplace=True)
        frame['StartDateTime'] = pd.to_datetime(frame['StartDateTime'])    
        return frame

if __name__ == "__main__":
    
    # create data source object
    tds = TestDataSource()
    data = tds.getSimulatedDataSeries()
    print(data)
    
    
    