# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 08:54:54 2020

@author: rhann
"""

import pandas as pd
import glob
import numpy as np
import os

# define primary and secondary data file folder path
path = 'C:/Users/rhann/Documents/GitHub/QAQC_Pipeline/sample_hourly_data/sd_data'

# get file names object
all_files = glob.glob(path + "/*.TXT")
        
for filename in all_files:
    
    # read each CSV file and only use specified columns
    # this read is for SM files
    print("reading file: " + filename)
    df = pd.read_csv(filename, index_col=None, header = None)
    
    # extract just the path portion of the globbed filename (head). The tail contains the actual filename.
    head, tail = os.path.split(filename)
    
    print(df[4])
    
    # fill all NaN values in AObs with concentratioin values from AObsAdj
    df[4] = df[4].fillna(df[7])
    
    # overwrite values in AObsAdj with NaN values
    df[7] = np.nan
    
    # write corrected dataframe to 'sd_data_c' subfolder as CSV file with '_c' appended to filename
    print("writing file: " + filename)
    df.to_csv(head + '/sd_data_c/' + tail[:-4] + '_c.TXT', header = False, index = False)
    
    print(df[4])
    
            
            
               