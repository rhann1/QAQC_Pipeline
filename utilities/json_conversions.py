# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:23:13 2020

@author: rhann
"""

import json
import pandas as pd
import pprint as pp




df = pd.read_csv("QA_metadata_sample.csv")


cols = ['UDL','LDL','MDL','PersistCount','PersistThresh','Anom1Thresh','Anom2Thresh','Anom3Thresh',
      'Anom4Thresh','PersistWindowSize','Anom1WindowSize','Anom2WindowSize','Anom3WindowSize',
      'Anom4WindowSize','Thresh1','Thresh2','Thresh3','Thresh4','WindowSize1','WindowSize2','WindowSize3','WindowSize4',
      'PerformCheck1','PerformCheck2','PerformCheck3','PerformCheck4','PerformCheck5','PerformCheck6','PerformCheck7']

j = (df.groupby(['StreamId','Description','DetectionLimitUOMcode','DurationCode','SamplingFreqCode'], as_index=False)
             .apply(lambda x: x[cols].to_dict('r'))
             .reset_index()
             .rename(columns={0:'QAConfigurationSettings'})
             .to_json(orient='records'))

j = json.loads(j)

pp.pprint(j)

# construction of simple two array JSON object
obj = {"configurations":None}

obj.update({"configurations":j})
print("") 
print("")
pp.pprint(obj)



