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




def ShouldQARun(token):
    response = requests.get('http://ab617-web-dev:8082/api/qa/GetQARunSettings', headers = {'Authorization': 'Bearer '+ token[0], 
                                                                                    'Content-Type': 'application/json; \
                                                                                    boundary=--------------------------651623359726858260475474'})
    enabled = json.loads(response.text)
    return enabled['QaSettings']['enabled']

def FetchQAScript(token):
    response = requests.get('http://ab617-web-dev:8082/api/qa/GetQAScript', headers = {'Authorization': 'Bearer '+ token[0], 
                                                                                    'Content-Type': 'application/json; \
                                                                                    boundary=--------------------------651623359726858260475474'})
    
    decoded_bytes = base64.b64decode(json.loads(response.text)['script']['QaScriptText'])
    with open("QC_core_test_DL.py", "wb") as download_file:
        download_file.write(decoded_bytes)

    return response.text        
    
if __name__ == "__main__":
    
    th = TokenHandler()
    token = th.getToken()
    
    if ShouldQARun(token) == 'true':
        print('run')
        scriptId = json.loads(FetchQAScript(token))['script']['QaScriptId']
        call(['python3', 'C:\Users\rhann\Documents\GitHub\QAQC_Pipeline\QC_core_test_DL.py'])
        # retrieve active QA script and ScriptId
        

