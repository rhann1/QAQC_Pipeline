# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:00:28 2020

@author: rhann
"""
import requests
import json

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
    
    # obtain script and QA metadata from /getqascript API endpoint
    response = requests.get('http://caqm-web-uat:8081/api/qa/getqascript', headers = {'Authorization': 'Bearer '+ token[0], 
                                                                                    'Content-Type': 'multipart/form-data; boundary=--------------------------651623359726858260475474'})
    # inspect response
    print(response.text.encode('utf8'))
    print(response.headers)

    print(response)



