# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:23:13 2020

@author: rhann
"""

# convert CSV QC config file to JSON object
import json
import pandas as pd
import pprint as pp

df = pd.read_csv("sample_QA_metadata/QA_metadata_sample_CCV_4s.csv")


cols = ['UDL', 'useUDL', 'LDL', 'useLDL', 'MDL', 'useMDL','PersistCount', 'usePCount', 'PersistThresh', 'usePThresh', 'Anom1Thresh', 'useA1Check', 
      'Anom2Thresh', 'useA2Check', 'Anom3Thresh', 'useA3Check', 'Anom4Thresh', 'useA4Check', 'PersistWindowSize','Anom1WindowSize','Anom2WindowSize',
	  'Anom3WindowSize', 'Anom4WindowSize','Thresh1','Thresh2','Thresh3','Thresh4','WindowSize1','WindowSize2','WindowSize3','WindowSize4',
      'PerformCheck1','PerformCheck2','PerformCheck3','PerformCheck4','PerformCheck5','PerformCheck6','PerformCheck7']

j = (df.groupby(['StreamSegmentId','Description', 'UOMId','SamplingDurationMinutes','SamplingFrequencySeconds'], as_index=False)
             .apply(lambda x: x[cols].to_dict('r'))
             .reset_index()
             .rename(columns={0:'QaConfigurationSettings'})
             .to_json(orient='records'))

j = json.loads(j)

pp.pprint(j)

# construction of simple two array JSON object
obj = {"configurations":None}

obj.update({"configurations":j})
print("") 
print("")
pp.pprint(obj)



