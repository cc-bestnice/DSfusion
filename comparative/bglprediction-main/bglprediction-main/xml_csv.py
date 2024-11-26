"""
Created in 2021

@author: Hoda Nemat (hoda.nemat@sheffield.ac.uk)
"""

import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime


def XMLtoCSV (data_type):

    d = {}
    
    subset_list=['training','testing']
    '''
    # Data related to challenge 2018
    Pid_list = ['559', '563', '570', '575', '588', '591']
    '''
    # Data related to challenge 2020
    Pid_list = ['540', '544', '552', '567', '584', '596']
    
    for k in range(len(subset_list)):
        d[subset_list[k]] = ['The Path of Data/OhioT1DM-2-' + subset_list[k] + '/'+ pID + '-ws-' + subset_list[k] + '.xml' for pID in Pid_list]
        
        for i in range(len(Pid_list)):
            tree = ET.parse(d[subset_list[k]][i])
            root = tree.getroot() 
            Glucose = root[0]
            GlucoseChildren = Glucose.getchildren()
                
            TS = []
            BGV = []
            
            for j in range(len(GlucoseChildren)):
                r = Glucose.getchildren()[j]
                ts = r.attrib['ts']
                TS.append(ts)
                value = r.attrib['value']
                BGV.append(value)
                
            TS_new = [datetime.strptime(d, '%d-%m-%Y %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S') for d in TS]
            Data = pd.Series(BGV, index=TS_new)
            Data.to_csv("%s_%s_%s.csv" % (data_type, Pid_list[i], subset_list[k]))  
            
XMLtoCSV (data_type='glucose_value')            
