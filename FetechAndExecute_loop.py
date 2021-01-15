# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:10:22 2020
@author: thelgestad
Last Modified on ___  
"""
import pandas as pd
import json
import requests
import sys
from token_handler import TokenHandler
from subprocess import call
import base64


api_host = 'ab617-web-dev:8082'
#api_host = 'caqm-web-uat:8082'


def ShouldQARun(token):
    response = requests.get('http://' + api_host + '/api/qa/GetQARunSettings', headers = {'Authorization': 'Bearer '+ token[0], 
                                                                                    'Content-Type': 'application/json; \
                                                                                    boundary=--------------------------651623359726858260475474'})
    enabled = json.loads(response.text)
    return enabled['QaSettings']['enabled']

def FetchQAScript(token):
    response = requests.get('http://' + api_host + '/api/qa/GetQAScript', headers = {'Authorization': 'Bearer '+ token[0], 
                                                                                    'Content-Type': 'application/json; \
                                                                                    boundary=--------------------------651623359726858260475474'})
    print("GetQaScript call resonded with " + str(response))
    
    decoded_bytes = base64.b64decode(json.loads(response.text)['script']['QaScriptText'])
    with open("QC_core_test_DL.py", "wb") as download_file:
        download_file.write(decoded_bytes)

    return response.text        
    
if __name__ == "__main__":
    
    th = TokenHandler()
    token = th.getToken()
    
    if ShouldQARun(token) == 'True':
        for i in range(10):
            print('run')
            response = json.loads(FetchQAScript(token))['script']['QaScriptId']
            print("executed FetchQAScript method to obtain QaScriptId = " + str(response))
            scriptId = json.loads(FetchQAScript(token))['script']['QaScriptId']
            print(scriptId)
            #call(['QC_core_test_DL.py', 'scriptId'], shell=True)
            import QC_core_test_DL
            QC_core_test_DL.main(True, scriptId)
            print('program complete')
            # retrieve active QA script and ScriptId
        
              
                
        
         