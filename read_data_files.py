# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:03:13 2020

@author: rhann
"""

import pandas as pd
import glob

class TestDataFileReader:
      
    def getDataFrameFromFiles(self, folder, subfolder):
        
        # define primary and secondary data file folder path
        path = folder+"/"+subfolder
        # get file names object
        all_files = glob.glob(path + "/*.csv")
        
        # create empty list and dataframe objects
        li = []
        frame = pd.DataFrame()
        
        # iterate over files and append data to the list object
        for filename in all_files:
            # read each CSV file and only use specified columns
            # this read is for SM files
            df = pd.read_csv(filename, index_col=None, header=0, usecols = ['StartDateTime', 'StreamId', 'MonitorId', 'Parameter', 'AObs'])
            # this read is for CCV files
            #df = pd.read_csv(filename, index_col=None, header=0, usecols = ['StartDateTime', 'MonitorId', 'Parameter', 'AObs'])
            li.append(df)
            # create concatenated dataframe object
            frame = pd.concat(li, axis=0, ignore_index=True)

        # creates test dataframe from multiple data files

        # rename MonitorId column to 'StreamId' to enforce compatibility with QC_Core required dataframe format (this won't be an issue with API calls)
        # this is required for all local test files that don't have a StreamID assigned (example: CCV data files)
        #frame.rename(columns={'MonitorId':'StreamId'}, inplace=True)
        frame['StartDateTime'] = pd.to_datetime(frame['StartDateTime'])   
        frame = frame[frame.columns[[0, 2, 1, 3, 4]]] # resequence column order in dataframe (example for Clarity data files)
        #frame = frame[frame.columns[[0, 2, 1, 3]]] # resequence column order in dataframe (example for CCV data files)
        #frame = frame.reindex(['StreamId', 'StartDateTime', 'Parameter', 'AObs'], axis=1) # alternative for resequencing order of columns
        # only keep these parameters (this can be a list)
        #frame = frame.loc[frame['Parameter'] == 88101]
        return frame

if __name__ == "__main__":
    
    # create data source object
    reader = TestDataFileReader()
    frame = reader.getDataFrameFromFiles('sample_subhourly_data', 'SM')
    #frame = reader.getDataFrameFromFiles('sample_subhourly_data', 'CCV')
    print(frame)
    

        
        