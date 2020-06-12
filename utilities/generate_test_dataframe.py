# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:27:22 2020

@author: rhann
"""

import pandas as pd
import json

jstring = '[{"IsSubHourly": false, "MeasurementId": 121888, "StreamSegmentId": 54, "IsCalculated": true, "QaProcessingLogId": 1, "QaConfigurationId": 3, "QF01": 1, "Qf02": 2, "Qf03": 3, "Qf04": 4, "Qf05": 5, "Qf06": 6, "Qf07": 7, "Qf08": 8, "Qf09": 9, "Qf10": 10, "QcValue01": 1.00001, "QcValue02": 2.00001, "QcValue03": 3.00001, "QcValue04": 4.00001, "QcValue05": 5.00001, "QcValue06": 6.00001, "QcValue07": 7.00001, "QcValue08": 8.00001, "QcValue09": 9.00001, "QcValue10": 10.00001, "QcText01": "pass1", "QcText02": "pass2", "QcText03": "pass3", "QcText04": "pass4", "QcText05": "pass5", "QcText06": "pass6", "QcText07": "pass7", "QcText08": "pass8", "QcText09": "pass9", "QcText10": "pass10"}, {"IsSubHourly": false, "MeasurementId": 143510, "StreamSegmentId": 54, "IsCalculated": true, "QaProcessingLogId": 1, "QaConfigurationId": 3, "QF01": 1, "Qf02": 2, "Qf03": 3, "Qf04": 4, "Qf05": 5, "Qf06": 6, "Qf07": 7, "Qf08": 8, "Qf09": 9, "Qf10": 10, "QcValue01": 1.00001, "QcValue02": 2.00001, "QcValue03": 3.00001, "QcValue04": 4.00001, "QcValue05": 5.00001, "QcValue06": 6.00001, "QcValue07": 7.00001, "QcValue08": 8.00001, "QcValue09": 9.00001, "QcValue10": 10.00001, "QcText01": "pass1", "QcText02": "pass2", "QcText03": "pass3", "QcText04": "pass4", "QcText05": "pass5", "QcText06": "pass6", "QcText07": "pass7", "QcText08": "pass8", "QcText09": "pass9", "QcText10": "pass10"}]'
jobs = json.loads(jstring)
QCFlagFrame = pd.DataFrame.from_dict(jobs, orient='columns')
    
print(QCFlagFrame)
    
    