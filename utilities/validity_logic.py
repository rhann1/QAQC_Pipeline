# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:40:03 2020

@author: rhann
"""

# test script to evaluate new QOverall logic for QC_Core
import pandas as pd
import numpy as np


def add_row(row):
    return row['a1'] + row['b1'] + row['c1']

def completeness(row):
    w = len(row)
    c = row[row >= 0].count()
    if c/w >= 0.75:
        return 1
    else:
        return 0
    
# create representative dataframe with status codes and NaN example
data = {'a1':[1,0,np.nan,-2], 'b1':[-2, 1,1,0], 'c1':[0, 1, 0, -2]}
f = pd.DataFrame(data, columns = ['a1', 'b1', 'c1'])

# replace NaN values with '-3' (not computed code)
f = f.fillna(-3)
 
# determine completeness of each example flag row (>= 0.75 is valid row)
f['valid'] = f.apply(completeness, axis = 1)
print(f)

# replace negative values with zero value for overall boolean calculation
f[f[['a1','b1','c1']] < 0] = 0

# compute overal quality flag using logical 'OR'  
f['overall'] = f[['a1', 'b1', 'c1']].any(axis = 1).astype(int)

print(f)
 
 
 
 
 