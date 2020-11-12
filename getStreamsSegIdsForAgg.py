# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 12:16:53 2020

@author: rhann
"""

f = computedQFlagFrame[['StartDateTime','StreamSegmentId']]
f
f.groupby(["StreamSegmentId", "StartDateTime"])

a = f.groupby(["StreamSegmentId", "StartDateTime"])

