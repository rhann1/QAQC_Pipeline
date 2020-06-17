# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:23:13 2020

@author: rhann
"""

import json
import pandas as pd




# construction of simple two array JSON object
obj = {"measurements":None, "configuration":None}

fragment1 = {"measurements":[
        {"StreamSegmentId": 1, "StartDateTime": 1, "AObs":1}, 
        {"StreamSegmentId": 2, "StartDateTime": 2, "Aobs":1}
        ]}

fragment2 =  {"configurations": [
    {
      "StreamSegmentId": 1,
      "SamplingFrequencyId": 1,
      "SamplingDurationId": 1,
      "UOMId": 1,
      "QaConfigurationSettings": [
        {
          "UDL": "500"
        },
        {
          "LDL": "0"
        }
      ]
    },
    {
      "StreamSegmentId": 2,
      "SamplingFrequencyId": 1,
      "SamplingDurationId": 1,
      "UOMId": 1,
      "QaConfigurationSettings": [
        {
          "UDL": "400"
        },
        {
          "LDL": "-2"
        }
      ]
    }
  ]}
      


obj.update({"measurements":fragment1['measurements']})
obj.update({"configuration":fragment2['configurations']})

# sample hello world dataframes used for object construction
df1 = pd.DataFrame({"foo": range(5), "bar": range(5, 10)})
df2 = pd.DataFrame({"foo": range(5), "bar": range(5, 10)}) 
 
j1 = df1.to_json(orient = 'records')
j2 = df2.to_json(orient = 'records')
print(j1)



