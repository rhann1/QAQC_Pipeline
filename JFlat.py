# -*- coding: utf-8 -*-
"""
Created on Wed May  6 21:58:20 2020

@author: rhann
"""

import pandas as pd
import re
import json
from flatten_json import flatten

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

    
    
    
    
    

