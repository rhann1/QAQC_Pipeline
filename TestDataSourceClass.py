# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:49:26 2020

@author: raifo
"""

import pandas as pd
import numpy as np
from read_data_files import TestDataFileReader

class TestDataSource:
    
    def __init__(self, mu = 15.0, sigma = 7):
        self.mu = mu
        self.sigma = sigma

    def getQAConfigDataFromFile(self, source):
        self.dataSource = source
        configFrame = pd.read_csv(self.dataSource)
        return configFrame
        
    def getSimulatedDataSeries(self):
        
        ###################################################################################################################
        # block for simulated data generation
        
        # generation of a simulated DateTime sequence
        # uncomment appropriate lines to select either hourly or subhourly data generation
        #tidx = pd.date_range('2000-01-01', '2000-12-31 23:00', freq='H') # StartDateTime index for hourly simulation
        tidx = pd.date_range('2000-01-01 00:00:00', '2000-01-01 23:00', freq='5min') # StartDateTime for sub-hourly simulation
        width = len(tidx) # determine width of time interval in records.  Used for selective placement of simulated outliers
    
        # generate simulated concentration series based on a normal distribution described by and mean (mu) and standard deviation (sigma)
        # s and s1 represents a separate series to be used with the two different StreamId groups
        s = np.random.normal(self.mu, self.sigma, width)
        s1 = np.random.normal(self.mu, self.sigma, width)
        
        # introduction of simulated anomalies into the data series
        s[int(width/2)] = 90 # moderate spike outlier
        s1[int(width/2)+100] = 275 # extreme spike outlier
        s1[int(width/4):int(width/4)+20] = 12.4 # insertion of 20 persistent values 
        
    
        # creation of time series dataframe
        frame = pd.DataFrame({'StartDateTime':tidx,'AObs':s})
        frame2 = pd.DataFrame({'StartDateTime':tidx,'AObs':s1})
        #frame2['StartDateTime'].iloc[8780] = pd.to_datetime('2000-12-31 20:30:00')    #test row for timediff check
        #frame2.drop(index=8774, inplace=True)

        # initialize two group IDs
        frame['StreamId']=0
        frame2['StreamId']=1
        # concatenate dataframes
        frame = frame.append(frame2, ignore_index=True)
        
        # end simulated data generation
        ######################################################################################################################
        
        ######################################################################################################################
        # block for alternatively reading from existing data files
        # comment out this section if simulated data generation is desired
        
        # select CSV formatted file to read
        #frame = pd.read_csv('sample_hourly_data/OCAP_SJV.csv')
        #frame = pd.read_csv('sample_subhourly_data/2020-02-08_1.csv', usecols = ['StartDateTime', 'MonitorId', 'Parameter', 'AObs'])
        
        # rename columns to match QC_Core module requirements
        #data.rename(columns={'DateTime':'StartDateTime', 'MeasuredValue':'AObs', 'SiteName':'StreamId'}, inplace=True)
        #frame.rename(columns={'MonitorId':'StreamId'}, inplace=True)
        #frame['StartDateTime'] = pd.to_datetime(frame['StartDateTime'])   
        #frame = frame[frame.columns[[0, 2, 1, 3]]] # resequence column order in dataframe
        #frame = frame.reindex(['StreamId', 'StartDateTime', 'Parameter', 'AObs'], axis=1) # alternative for resequencing order of columns
        #frame = frame.loc[frame['Parameter'] == 88101]
        
        # create data source object
        reader = TestDataFileReader()
        #frame = reader.getDataFrameFromFiles('sample_subhourly_data', 'CCV')
        #frame = reader.getDataFrameFromFiles('sample_subhourly_data', 'SM')
        frame = reader.getDataFrameFromFiles('sample_subhourly_data', 'SM')
        #frame = frame.loc[frame['Parameter'] == 85101]
        
        # end read from CSV file block
        #####################################################################################################################
        
        return frame

if __name__ == "__main__":
    
        # create data source object
        tds = TestDataSource()
        data = tds.getSimulatedDataSeries()
        configData = tds.getQAConfigDataFromFile('sample_QA_metadata/QA_metadata_sample_2.csv')
        print(data)
    
    
    