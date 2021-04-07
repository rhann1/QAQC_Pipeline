# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:00:28 2020

@author: rhann
"""
import requests
import json
import pandas as pd
from pandas.io.json import json_normalize
import base64
from datetime import datetime
import sys
import numpy as np



class TokenHandler:
    
    def __init__(self):
        self.token = None
        
    def getToken(self):
        # url of IDM authority
        url = "https://idm-dev.arb.ca.gov/connect/token"

        # parameters for token generation
        payload = {
                "Address":"disco.TokenEndpoint",
                "client_id":"AQViewDockerClient",
                "client_secret":"4qv13wd0ckercl13nt!!",
                "grant_type":"client_credentials", 
                "Scope":"AQViewQAAPI.full_access"
                }

        # POST request headers for IDM endpoint
        headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
                }

        # perform request and capture response
        response = requests.post(url, headers=headers, data=payload)
        status = response.status_code
        error_object = response.raise_for_status()
        

        # capture token from IDM response
        res = json.loads(response.text.encode('utf8'))
        token = res['access_token']
        return token,status,error_object
    

if __name__ == "__main__":
    
    th = TokenHandler()
    token = th.getToken()

    # inspect token
    print(token[0])

    
    #enter identifiers for QC Flags groups that should be cleared
    
    jobj = {
            'IsSubhourly': False, 
            'QaScriptIds': [],
            'StreamSegmentIds':[620]
            }
    
    response = requests.post('http://caqm-web:8082/api/qa/ClearIsCalculatedFlag', headers = {'Authorization': 'Bearer '+ token[0], 
                                                                                    'Content-Type': 'application/json; \
                                                                                    boundary=--------------------------651623359726858260475474'}, \
                                                                                    json=jobj)
    print(response)


        

    
    
    
    
    
    
    
    
    



