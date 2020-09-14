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

""

# code for inserting encoded QC Core script into the database

#Base64 encode QA script
with open("../QC_core_test_DL.py", "rb") as script_file:
    encoded_script = base64.b64encode(script_file.read()).decode('utf-8')
    print(encoded_script)

# create JSON payload    
payload = {'script': encoded_script}
# issue request to API
"""
response = requests.post('http://ab617-web-dev:8082/api/qa/PutQAScript', headers = {'Authorization': 'Bearer '+ token[0], 
                                                                                    'Content-Type': 'application/json; \
                                                                                    boundary=--------------------------651623359726858260475474'}, \
                                                                                    json=payload)

print(response)
"""


# code for inserting QC metadata into the database

# read QC metadata into dataframe
#df = pd.read_csv("../sample_QA_metadata/QA_metadata_sample_SC_2s.csv")
#df = pd.read_csv("../sample_QA_metadata/SM_QCMetadata_2s.csv")
df = pd.read_csv("../sample_QA_metadata/QC_SM_DEV_full.csv")

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
jdata = json.dumps(obj)

# need to insert API request here
response = requests.post('http://ab617-web-dev:8082/api/qa/PutStreamConfigurations', headers = {'Authorization': 'Bearer '+ token[0], 
                                                                                'Content-Type': 'application/json; \
                                                                                boundary=--------------------------651623359726858260475474'}, \
                            

                                                                                data=jdata)

print(response)






