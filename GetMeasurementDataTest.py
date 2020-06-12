# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:00:28 2020

@author: rhann
"""
import requests
import json
from pandas.io.json import json_normalize
import base64

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
    response = requests.post('http://ab617-web-dev:8082/api/qa/GetMeasurementDataForQA', headers = {'Authorization': 'Bearer '+ token[0], 
                                                                                    'Content-Type': 'application/json; \
                                                                                    boundary=--------------------------651623359726858260475474'}, \
                                                                                    json={"IsSubHourly": "False", "IntervalHours": "1", "MaxNumberOfStreams": "1000"})
    # inspect response
    print(response.text.encode('utf8'))
    print(response.headers)
    jstring = '[{"IsSubHourly": false, "MeasurementId": 121888, "StreamSegmentId": 54, "IsCalculated": true, "QaProcessingLogId": 1, "QaConfigurationId": 3, "QF01": 1, "Qf02": 2, "Qf03": 3, "Qf04": 4, "Qf05": 5, "Qf06": 6, "Qf07": 7, "Qf08": 8, "Qf09": 9, "Qf10": 10, "QcValue01": 1.00001, "QcValue02": 2.00001, "QcValue03": 3.00001, "QcValue04": 4.00001, "QcValue05": 5.00001, "QcValue06": 6.00001, "QcValue07": 7.00001, "QcValue08": 8.00001, "QcValue09": 9.00001, "QcValue10": 10.00001, "QcText01": "pass1", "QcText02": "pass2", "QcText03": "pass3", "QcText04": "pass4", "QcText05": "pass5", "QcText06": "pass6", "QcText07": "pass7", "QcText08": "pass8", "QcText09": "pass9", "QcText10": "pass10"}, {"IsSubHourly": false, "MeasurementId": 143510, "StreamSegmentId": 54, "IsCalculated": true, "QaProcessingLogId": 1, "QaConfigurationId": 3, "QF01": 1, "Qf02": 2, "Qf03": 3, "Qf04": 4, "Qf05": 5, "Qf06": 6, "Qf07": 7, "Qf08": 8, "Qf09": 9, "Qf10": 10, "QcValue01": 1.00001, "QcValue02": 2.00001, "QcValue03": 3.00001, "QcValue04": 4.00001, "QcValue05": 5.00001, "QcValue06": 6.00001, "QcValue07": 7.00001, "QcValue08": 8.00001, "QcValue09": 9.00001, "QcValue10": 10.00001, "QcText01": "pass1", "QcText02": "pass2", "QcText03": "pass3", "QcText04": "pass4", "QcText05": "pass5", "QcText06": "pass6", "QcText07": "pass7", "QcText08": "pass8", "QcText09": "pass9", "QcText10": "pass10"}]'
    jobs = json.loads(jstring)
    QCFlagFrame = pd.DataFrame.from_dict(jobs, orient='columns')
    

    """
    measurements = json.loads(response.text)[0]['MeasurementData']
    measurementsFrame = json_normalize(measurements)
    measurementsFrame.drop(['Aobs', 'AobsAdj', 'Bobs', 'BobsAdj', 'Cobs', 'CobsAdj', 'ConvUomid', 'StartDateTime', 'Uomid'], axis=1, inplace=True)
    print(measurementsFrame)
    measurementsFrame['IsCalculated'] = None
    measurementsFrame['QaProcessingLogId'] = None
    measurementsFrame['QaConfigurationId'] = None
    measurementsFrame['Qf01'] = None   
    measurementsFrame['Qf01'] = None
    measurementsFrame['Qf02'] = None
    measurementsFrame['Qf03'] = None
    measurementsFrame['Qf04'] = None
    measurementsFrame['Qf05'] = None
    measurementsFrame['Qf06'] = None
    measurementsFrame['Qf07'] = None
    measurementsFrame['Qf08'] = None
    measurementsFrame['Qf09'] = None
    measurementsFrame['Qf10'] = None
    measurementsFrame['QcValue01'] = None
    measurementsFrame['QcValue02'] = None
    measurementsFrame['QcValue03'] = None
    measurementsFrame['QcValue04'] = None
    measurementsFrame['QcValue05'] = None
    measurementsFrame['QcValue06'] = None
    measurementsFrame['QcValue07'] = None
    measurementsFrame['QcValue08'] = None
    measurementsFrame['QcValue09'] = None
    measurementsFrame['QcValue10'] = None
    measurementsFrame['QcText01'] = None
    measurementsFrame['QcText02'] = None
    measurementsFrame['QcText03'] = None
    measurementsFrame['QcText04'] = None
    measurementsFrame['QcText05'] = None
    measurementsFrame['QcText06'] = None
    measurementsFrame['QcText07'] = None
    measurementsFrame['QcText08'] = None
    measurementsFrame['QcText09'] = None
    measurementsFrame['QcText10'] = None
        
    measurementsFrame['Qf01'][measurementsFrame['StreamSegmentId'] == 54] = 0
    measurementsFrame['QaConfigurationId'][measurementsFrame['StreamSegmentId'] == 54] = 3
    measurementsFrame['QaProcessingLogId'][measurementsFrame['StreamSegmentId'] == 54] = 16
    measurementsFrame['IsCalculated'][measurementsFrame['StreamSegmentId'] == 54] = True
    measurementsFrame['IsCalculated'][measurementsFrame['StreamSegmentId'] == 54] = False
    measurementsFrame['IsSubHourly'][measurementsFrame['StreamSegmentId'] == 54] = False
    measurementsFrame['MeasurementId'][measurementsFrame['StreamSegmentId'] == 54] = 121888
    measurementsFrame['QaProcessingLogId'][measurementsFrame['StreamSegmentId'] == 54] = 1
    measurementsFrame['Qf01'][measurementsFrame['StreamSegmentId'] == 54] = 1
    measurementsFrame['Qf02'][measurementsFrame['StreamSegmentId'] == 54] = 2
    measurementsFrame['Qf03'][measurementsFrame['StreamSegmentId'] == 54] = 3
    measurementsFrame['Qf04'][measurementsFrame['StreamSegmentId'] == 54] = 4
    measurementsFrame['Qf05'][measurementsFrame['StreamSegmentId'] == 54] = 5
    measurementsFrame['Qf06'][measurementsFrame['StreamSegmentId'] == 54] = 6
    measurementsFrame['Qf07'][measurementsFrame['StreamSegmentId'] == 54] = 7
    measurementsFrame['Qf08'][measurementsFrame['StreamSegmentId'] == 54] = 8
    measurementsFrame['Qf09'][measurementsFrame['StreamSegmentId'] == 54] = 9
    measurementsFrame['Qf10'][measurementsFrame['StreamSegmentId'] == 54] = 10
    measurementsFrame['QcValue01'][measurementsFrame['StreamSegmentId'] == 54] = 1.00001
    measurementsFrame['QcValue02'][measurementsFrame['StreamSegmentId'] == 54] = 2.00001
    measurementsFrame['QcValue03'][measurementsFrame['StreamSegmentId'] == 54] = 3.00001
    measurementsFrame['QcValue04'][measurementsFrame['StreamSegmentId'] == 54] = 4.00001
    measurementsFrame['QcValue05'][measurementsFrame['StreamSegmentId'] == 54] = 5.00001
    measurementsFrame['QcValue06'][measurementsFrame['StreamSegmentId'] == 54] = 6.00001
    measurementsFrame['QcValue07'][measurementsFrame['StreamSegmentId'] == 54] = 7.00001
    measurementsFrame['QcValue08'][measurementsFrame['StreamSegmentId'] == 54] = 8.00001
    measurementsFrame['QcValue09'][measurementsFrame['StreamSegmentId'] == 54] = 9.00001
    measurementsFrame['QcValue10'][measurementsFrame['StreamSegmentId'] == 54] = 10.00001
    measurementsFrame['QcText01'][measurementsFrame['StreamSegmentId'] == 54] = "pass1"
    measurementsFrame['QcText02'][measurementsFrame['StreamSegmentId'] == 54] = "pass2"
    measurementsFrame['QcText03'][measurementsFrame['StreamSegmentId'] == 54] = "pass3"
    measurementsFrame['QcText04'][measurementsFrame['StreamSegmentId'] == 54] = "pass4"
    measurementsFrame['QcText05'][measurementsFrame['StreamSegmentId'] == 54] = "pass5"
    measurementsFrame['QcText06'][measurementsFrame['StreamSegmentId'] == 54] = "pass6"
    measurementsFrame['QcText07'][measurementsFrame['StreamSegmentId'] == 54] = "pass7"
    measurementsFrame['QcText08'][measurementsFrame['StreamSegmentId'] == 54] = "pass8"
    measurementsFrame['QcText09'][measurementsFrame['StreamSegmentId'] == 54] = "pass9"
    measurementsFrame['QcText10'][measurementsFrame['StreamSegmentId'] == 54] = "pass10"
    measurementsFrame['IsCalculated'][measurementsFrame['StreamSegmentId'] == 54] = True
    measurementsFrame = measurementsFrame.head(1)
    measurementsJSON = measurementsFrame.to_json(orient = 'records')
    print(measurementsJSON)
    payload = measurementsJSON
    
    response = requests.post('http://ab617-web-dev:8082/api/qa/PutMeasurementDataForQA', headers = {'Authorization': 'Bearer '+ token[0], 
                                                                                    'Content-Type': 'application/json; \
                                                                                    boundary=--------------------------651623359726858260475474'}, \
                                                                                    json=payload)
    # inspect response
    print(response.text.encode('utf8'))
    print(response.headers)
"""    
"""    #Base64 encode QA script
    with open("QC_core_test.py", "rb") as script_file:
        encoded_script = base64.b64encode(script_file.read()).decode('utf-8')
        print(encoded_script)
    
    payload = {'script': encoded_script}
    response = requests.post('http://ab617-web-dev:8082/api/qa/PutQAScript', headers = {'Authorization': 'Bearer '+ token[0], 
                                                                                    'Content-Type': 'application/json; \
                                                                                    boundary=--------------------------651623359726858260475474'}, \
                                                                                    json=payload)
    print(response)
    
    response = requests.post('http://ab617-web-dev:8082/api/qa/PutQAScript', headers = {'Authorization': 'Bearer '+ token[0], 
                                                                                    'Content-Type': 'application/json; \
                                                                                    boundary=--------------------------651623359726858260475474'}, \
                                                                                    json=payload)
    
    response = requests.get('http://ab617-web-dev:8082/api/qa/GetQAScript', headers = {'Authorization': 'Bearer '+ token[0], 
                                                                                    'Content-Type': 'application/json; \
                                                                                    boundary=--------------------------651623359726858260475474'})
    
    decoded_bytes = base64.b64decode(json.loads(response.text)['script']['QaScriptText'])
    with open("QC_core_test_DL.py", "wb") as download_file:
        download_file.write(decoded_bytes)
    
    
    """  
    
    jstring = '[{"IsSubHourly": false, "MeasurementId": 121888, "StreamSegmentId": 54, "IsCalculated": true, "QaProcessingLogId": 1, "QaConfigurationId": 3, "QF01": 1, "Qf02": 2, "Qf03": 3, "Qf04": 4, "Qf05": 5, "Qf06": 6, "Qf07": 7, "Qf08": 8, "Qf09": 9, "Qf10": 10, "QcValue01": 1.00001, "QcValue02": 2.00001, "QcValue03": 3.00001, "QcValue04": 4.00001, "QcValue05": 5.00001, "QcValue06": 6.00001, "QcValue07": 7.00001, "QcValue08": 8.00001, "QcValue09": 9.00001, "QcValue10": 10.00001, "QcText01": "pass1", "QcText02": "pass2", "QcText03": "pass3", "QcText04": "pass4", "QcText05": "pass5", "QcText06": "pass6", "QcText07": "pass7", "QcText08": "pass8", "QcText09": "pass9", "QcText10": "pass10"}, {"IsSubHourly": false, "MeasurementId": 143510, "StreamSegmentId": 54, "IsCalculated": true, "QaProcessingLogId": 1, "QaConfigurationId": 3, "QF01": 1, "Qf02": 2, "Qf03": 3, "Qf04": 4, "Qf05": 5, "Qf06": 6, "Qf07": 7, "Qf08": 8, "Qf09": 9, "Qf10": 10, "QcValue01": 1.00001, "QcValue02": 2.00001, "QcValue03": 3.00001, "QcValue04": 4.00001, "QcValue05": 5.00001, "QcValue06": 6.00001, "QcValue07": 7.00001, "QcValue08": 8.00001, "QcValue09": 9.00001, "QcValue10": 10.00001, "QcText01": "pass1", "QcText02": "pass2", "QcText03": "pass3", "QcText04": "pass4", "QcText05": "pass5", "QcText06": "pass6", "QcText07": "pass7", "QcText08": "pass8", "QcText09": "pass9", "QcText10": "pass10"}]'
  
    
     
     
    
    
        
    
    
    
    
    
    
    
    
    



