# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:10:22 2020

@author: rhann
"""
import pandas as pd
import re
import json
from pprint import pprint
from flatten_json import flatten
from token_handler import TokenHandler


class DataHandler:
    
    def getData(self, payload): # method to call measurement data API (# 13 on QC Pipeline sequence diagram)
                
        # 1. call TokenHandler() and get access token (sequence #14)
        # 2. prepare REST request w/ access token
        # 3. call API and get JSON payload containing measurement data and QA metadata (sequence #16)
        # 4. convert JSON object to QAConfig dataframe and measurement dataframe (sequence #18)
        
        # inner function to extract measurement and configuration data and convert to dataframe objects (# 18 on QC Pipeline sequence diagram)        
        def ConvertJSONtoDataframes(payload):
    
            jMeasurementData   = payload['measurements']
            jConfigurationData = payload['configuration']
    
            # need to flatten nested JSON configuration data
            jConfigurationData = json.dumps([flatten(j) for j in jConfigurationData])
        
            # prune QAConfig prefix from nested keys and convert object to dataframe
            jPruned = re.sub(r'QaConfigurationSettings_[\d]+_', '', jConfigurationData)
            configurationFrame = pd.DataFrame(json.loads(jPruned))
    
            # convert measurement data object to dataframe
            measurementFrame = pd.DataFrame(jMeasurementData)
    
            return measurementFrame, configurationFrame
        
        frames = ConvertJSONtoDataframes(payload)
        return frames
    
    def putQCFlagData(self, QFrame):    # method to POST computed QC flags to  the data API (# 25 on QC Pipeline diagram)
        
        QCFlagData = self.QFrame        # QFrame is the dataframe generated by the QC_Core module that contains the computed QC Flags for each record
        
        # call TokenHandler and get access token (sequence #22)
        # convert dataframe to JSON (sequence #24)
        # prepare REST request w/ payload and access token
        # call putQAFlagData API and POST JSON payload to API (sequence #25)
        
    def putHourlyAverageData(self, averagesData):    # method to POST computed hourly averages to data API (not on sequence diagram but will occur after # 26)
        
        AveragesDataframe = self.averagesData            # averagesData is the dataframe generated by the QC_Core module that contains the aggregated subhourly data
        # convert dataframe to JSON object
        # call TokenHandler and get access token
        # prepare REST request w/ payload and access token
        # call putHourlyAverages API and POST JSON payload to API

#######################################################################################################################################################
        
if __name__ == "__main__":
    
    # test of JSON to dataframe conversion feature of 'getData()" method
    # read test JSON object from file (not needed in production)
    with open('sample_json_payload/simple_payload.json') as f:
        payload = json.load(f)

    pprint(payload)
    print('')
    dh = DataHandler()
    frames = dh.getData(payload)
    
    print(frames[0])
    print('')
    print(frames[1])
        
    
        
        
        