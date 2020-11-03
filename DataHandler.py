# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:10:22 2020
@author: thelgestad
Last Modified on ___
"""
import pandas as pd
import re
import json
import time
import requests
import sys
import pprint
from flatten_json import flatten
from token_handler import TokenHandler as th
from pandas.io.json import json_normalize

class DataHandler:
    
    def getData(self, IsSubHourlyVal, IntervalHoursVal, MaxNumberOfStreamsVal):
        print("maximum number of streams = " + str(MaxNumberOfStreamsVal))

        def PreparePayload(IsSubHourlyVal, IntervalHoursVal, MaxNumberOfStreamsVal):
            
            GetDataSettingsPayload = {"IsSubHourly": IsSubHourlyVal, "IntervalHours": IntervalHoursVal, "MaxNumberOfStreams": MaxNumberOfStreamsVal}
            return(GetDataSettingsPayload)

        #call TokenHandler() and get access token
        def RequestToken():
    
            token = th.getToken(self)
            
            if token[1] != 200:
                sys.exit("Token retrieval error: " + token[2])
            else:
                return(token[0])

        def GetMeasurementandConfig(GetDataSettingsPayload, token): 
                        
             try:
                 MeasurementDataPayload = requests.post('http://ab617-web-dev:8082/api/qa/GetMeasurementDataForQA', json = GetDataSettingsPayload, headers = {'Authorization': 'Bearer '+ token, 'Content-Type': 'application/json; boundary=--------------------------651623359726858260475474'})

             except requests.exceptions.RequestException as e:  
                 raise SystemExit(e)

             return(MeasurementDataPayload )

        #inner function to extract measurement and configuration data and convert to dataframe objects (# 18 on QC Pipeline sequence diagram)        
        def ConvertJSONtoDataframe(payload):
                        
            print(payload.text)
            
            jConfigurations = json.loads(payload.text)[0]['Configurations']
            jMeasurementData = json.loads(payload.text)[0]['MeasurementData']
            
            measurementFrame = pd.DataFrame(jMeasurementData)
                       
            # prune QAConfig prefix from nested keys and convert object to dataframe
            jConfigurations = json.dumps([flatten(j) for j in jConfigurations])
            jPruned = re.sub(r'Settings_[\d]+_', '', jConfigurations)
            configurationFrame = pd.DataFrame(json.loads(jPruned))
            
            return (measurementFrame, configurationFrame)

        GetDataSettingsPayload = PreparePayload(IsSubHourlyVal, IntervalHoursVal, MaxNumberOfStreamsVal)                                                                            
        Token = RequestToken()
        MeasurementPayload = GetMeasurementandConfig(GetDataSettingsPayload, Token)
        DfsForProcessing = ConvertJSONtoDataframe(MeasurementPayload)
        
        # added return statement for getData() method (RHann)
        return DfsForProcessing[0], DfsForProcessing[1]
    
    def GetBatchId(self, QaScriptId, QaProcessingStart, BatchSize):
    
        def RequestToken():
    
            token = th.getToken(self)
            
            if token[1] != 200:
                sys.exit("Token retrieval error: " + token[2])
            else:
                return(token[0])

        jobj = {'QaScriptId' : QaScriptId, 
                'QaProcessingStart': str(QaProcessingStart), # assume current datetime was generated by datetime.now() 
                'QaProcessingEnd': None, 
                'BatchSize': BatchSize} 

        token = RequestToken()
                    
        try:
       
            response = requests.post('http://ab617-web-dev:8082/api/qa/PutQAProcessingProgress', headers = {'Authorization': 'Bearer '+ token, 
                                                                                    'Content-Type': 'application/json; \
                                                                                    boundary=--------------------------651623359726858260475474'}, \
                                                                                    json=jobj)
        except requests.exceptions.RequestException as e:  
            raise SystemExit(e)

        return(response.text)
        
    def PutProcessingLog(self, QaProcessingLogId, QaProcessingEnd, successfulRecordCount):
        
        def RequestToken():
    
            token = th.getToken(self)
            
            if token[1] != 200:
                sys.exit("Token retrieval error: " + token[2])
            else:
                return(token[0])
                
        jobj = {'QaProcessingLogId': str(QaProcessingLogId), 
                'QaProcessingEnd': str(QaProcessingEnd), 
                'SuccessfulRecordCount': str(successfulRecordCount)}
        
        token = RequestToken()
                    
        try:
            response = requests.post('http://ab617-web-dev:8082/api/qa/PutQAProcessingProgress', headers = {'Authorization': 'Bearer '+ token, 
                                                                                    'Content-Type': 'application/json; \
                                                                                    boundary=--------------------------651623359726858260475474'}, \
                                                                                    json=jobj)
                    
        except requests.exceptions.RequestException as e:  
            print("error occurs in sending process log")
            raise SystemExit(e)
        
        print("sent QaProcessingLogId " + str(QaProcessingLogId))
        print("received response " + str(response))
        return(response.text)
        
    def PutComputedFlags(self, computedFlags):
        
        def RequestToken():
    
            token = th.getToken(self)
            
            if token[1] != 200:
                sys.exit("Token retrieval error: " + token[2])
            else:
                return(token[0])
                
        payload = {'DataToPut': None}
        jobj = computedFlags.to_json(orient = 'records')
        jobj = json.loads(jobj)
        payload.update({'DataToPut': jobj})
        
        token = RequestToken()
        #print(payload)
        
                    
        try:
            start = time.time()
            response = requests.post('http://ab617-web-dev:8082/api/qa/PutMeasurementDataForQA', headers = {'Authorization': 'Bearer '+ token, 
                                                                                    'Content-Type': 'application/json; \
                                                                                    boundary=--------------------------651623359726858260475474'}, \
                                                                                    json=payload)
            end = time.time()
            print("QC flag insertion time = " + str(end - start))
            print(payload)
            print(response)
            print(response.text)
        except requests.exceptions.RequestException as e: 
            print("error occured sending QC flags")
            raise SystemExit(e)
        
        print("sent computed QC flags")
        print("received response " + str(response))         
        return(response.text)
                
        
        
        

        
 #       DfsForProcessing[0].to_csv(r"C:\Users\tahelges\Desktop\payload0.csv", index = False)
  #      DfsForProcessing[1].to_csv(r"C:\Users\tahelges\Desktop\payload1.csv", index = False)

        
    #def Processdata(dfs):
        #call QC core

    #def putData(dfs) 
        #convert dataframes to JSON
        #put JSON into db
    
        
#######################################################################################################################################################
        
if __name__ == "__main__":
    
    from datetime import datetime
    dh = DataHandler()
    dfs = dh.getData("True", "1", 4)
    batchId = dh.GetBatchId(1, datetime.now(), 16000 )
    #dh.ProcessData(dfs)
    #dh.putData()
    #print(dfs[0])
    #print(dfs[1])

""" 
        def putQCFlagData(self, QFrame):    # method to POST computed QC flags to  the data API (# 25 on QC Pipeline diagram)
        
        QCFlagData = self.QFrame        # QFrame is the dataframe generated by the QC_Core module that contains the computed QC Flags for each record
    
        
    def putHourlyAverageData(self, averagesData):    # method to POST computed hourly averages to data API (not on sequence diagram but will occur after # 26)
        
        AveragesDataframe = self.averagesData            # averagesData is the dataframe generated by the QC_Core module that contains the aggregated subhourly data """