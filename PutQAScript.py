# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 22:34:07 2020

@author: rhann
"""


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
    api_host = 'caqm-web-uat:8082'    # change is value to point to a different host

    # inspect token
    print(token[0])
        
    #Base64 encode QA script
    with open("QC_core_test_DL.py", "rb") as script_file:
        encoded_script = base64.b64encode(script_file.read()).decode('utf-8')
        print(encoded_script)
    
    payload = {'script': encoded_script}
    response = requests.post('http://' + api_host + '/api/qa/PutQAScript', headers = {'Authorization': 'Bearer '+ token[0], 
                                                                                    'Content-Type': 'application/json; \
                                                                                    boundary=--------------------------651623359726858260475474'}, \
                                                                                    json=payload)
    print(response)
    
    if response.status_code == 200:
        print("Script uploaded successfully")
        
        
    
