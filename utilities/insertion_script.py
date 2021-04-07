# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 08:16:13 2020

@author: rhann
"""

import requests
import json
import pandas as pd
import base64
import pprint as pp
from token_handler import TokenHandler


# get token from token handler
th = TokenHandler()
token = th.getToken()

api_host = 'caqm-web-uat:8082'
#api_host = 'ab617-web-dev:8082'
api_host = 'caqm-web:8082'



# code for inserting QC metadata into the database

# read QC metadata into dataframe
#df = pd.read_csv("../sample_QA_metadata/QA_metadata_sample_SC_2s.csv")
#df = pd.read_csv("../sample_QA_metadata/SM_QCMetadata_2s.csv")
#df = pd.read_csv("../sample_QA_metadata/QC_Metadata_SM_UAT_merged.csv")
#df = pd.read_csv("../sample_QA_metadata/QC_SM_DEV_full_merged.csv")
df = pd.read_csv("../sample_QA_metadata/SC_QC_Config_2_20210308.csv")

# convert CSV QC config file to JSON object

# define column list for extensible key-value pairs
cols = ['computeQuality', 'UDL', 'useUDL', 'LDL', 'useLDL', 'MDL', 'useMDL','PersistCount', 'usePCount', 'PersistThresh', 'usePThresh', 'useAdj', 'Anom1Thresh', 'useA1Check', 
      'Anom2Thresh', 'useA2Check', 'Anom3Thresh', 'useA3Check', 'Anom4Thresh', 'useA4Check', 'Anom7Thresh', 'useA7Check', 'PersistWindowSize','Anom1WindowSize','Anom2WindowSize',
	  'Anom3WindowSize', 'Anom4WindowSize','Thresh1','Thresh2','Thresh3','Thresh4','WindowSize1','WindowSize2','WindowSize3','WindowSize4',
      'PerformCheck1','PerformCheck2','PerformCheck3']

df[cols] = df[cols].astype(str)

# created nested objects of extensible key-value pairs
j = (df.groupby(['StreamSegmentId', 'UOMId','SamplingDurationMinutes','SamplingFrequencySeconds'], as_index=False)
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

# convert nested array into nested list of objects for API compatibility
for c in obj['configurations']:
    s = c['QaConfigurationSettings'][0]
    c.update({"QaConfigurationSettings": None})
    c.update({"QaConfigurationSettings": s})
    #c.update({"QaConfigurationId": None})
    pp.pprint(c)
    print(' ')
    
    
pp.pprint(obj)
# convert JSON object to string  
#obj =  {"configurations":[{"QaConfigurationId":170,"StreamSegmentId":None,"SamplingFrequencySeconds":None,"SamplingDurationMinutes":None,"UOMId":None,"QaConfigurationSettings":None}]}
jdata = json.dumps(obj)

# need to insert API request here
response = requests.post('http://' + api_host + '/api/qa/PutStreamConfigurations', headers = {'Authorization': 'Bearer '+ token[0], 
                                                                                'Content-Type': 'application/json; \
                                                                                boundary=--------------------------651623359726858260475474'}, \
                            

                                                                                data=jdata)

print(response)

if response.status_code == 200:
    print("QC Configuration uploaded successfully")
else:
    print("QC Configuration upload failed")
    
    
    

    
    
    







