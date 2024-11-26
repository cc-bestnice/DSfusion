
"""
Created in 2021

@author: Hoda Nemat (hoda.nemat@sheffield.ac.uk)
"""

import pandas as pd
from pandas import read_csv
from numpy import array
import numpy as np
from datetime import datetime
import pickle as pickle
from copy import copy

def FindMissing_interp (seq, subset):
    
    withoutsec_str = [datetime.strptime(str(d), '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M') for d in seq.index]
    withoutsec_list = [datetime.strptime(d, '%Y-%m-%d %H:%M') for d in withoutsec_str]
    withoutsec = pd.DatetimeIndex(withoutsec_list)
 
    rounded_withoutsec = [d - pd.Timedelta(minutes=d.minute % 5) for d in withoutsec] 
    seq.index = rounded_withoutsec
    seq=seq.loc[~seq.index.duplicated(keep='first')]
    
    seq_new = pd.Series(seq.values[0], index=[seq.index[0]]) #this will include missing data


    for i in range(1, len(seq)):
        
        time_delta =  pd.Timestamp(seq.index[i]) - pd.Timestamp(seq.index[i-1])

        if time_delta > pd.Timedelta('9min'): 

            seq_nan = seq.reindex(pd.date_range(start=seq.index[i-1],end=seq.index[i], freq='5min', closed='right'))
    
            seq_new=pd.concat([seq_new,seq_nan])
    
        else:
    
            real_row=pd.Series(seq.values[i],index=[seq.index[i]])
            seq_new=pd.concat([seq_new, real_row])
    
    
    nan_mask = np.isnan(seq_new) 
    
    if subset == "training":
        interpolated = seq_new.interpolate(limit_direction='both')
    if subset == "testing":
        interpolated = seq_new.interpolate(limit_direction='forward')




    return interpolated, nan_mask

def split_XY (sequence, n_steps_in, n_steps_out):
    
    nan_mask = sequence['nan_mask']
    del sequence['nan_mask']
    seq = sequence['BG']
    
    X = []
    Y = []
    X_NanMask = []
    Y_NanMask = []

    
    for i in range(len(seq)):
        
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        
        seq_x = seq[i:end_ix]
        seq_y = seq[end_ix:out_end_ix]
        
        X_withNan = nan_mask[i:end_ix]
        Y_withNan = nan_mask[end_ix:out_end_ix]
    

        if Y_withNan[-1] == False:
           
           
            X.append(seq_x)
            Y.append(seq_y)
            
            
            X_NanMask.append(X_withNan)
            Y_NanMask.append(Y_withNan)
            
       
    XNanMask_array = array(X_NanMask)       
    YNanMask_array = array(Y_NanMask) 
       
    X_array = array(X)       
    Y_array = array(Y)
    
    return X_array, Y_array, XNanMask_array, YNanMask_array






    
d = {}
Pid_list = ['540', '544', '552','559', '563', '567', '570', '575', '584', '588', '591', '596']   
histories = ['6','12','18','24']
horizons = ['6','12']
subset =['training','testing']

for i in range(len(Pid_list)):    
    for s in range(len(histories)):
        for h in range(len(horizons)):
            for k in range(len(subset)):

                d[Pid_list[i]+'_'+subset[k]] = read_csv('glucose_value_' + Pid_list[i] + '_' + subset[k] + '.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
                seq = copy(d[Pid_list[i]+'_'+subset[k]])
                seq_glucose_value, nan_mask = FindMissing_interp(seq, subset[k])
                
                BG = seq_glucose_value.values
                TS_BG = seq_glucose_value.index
                Data = pd.DataFrame({'BG': BG, 'nan_mask': nan_mask.values}, index=TS_BG)
                Data.to_csv('%s_%s.csv' % (subset[k], Pid_list[i]))    
                
                Data = Data[:]  
                d['X_{}_{}_{}'.format(subset[k],histories[s],horizons[h])],d['Y_{}_{}_{}'.format(subset[k],histories[s],horizons[h])], d['XNanMask_{}_{}_{}'.format(subset[k],histories[s],horizons[h])], d['YNanMask_{}_{}_{}'.format(subset[k],histories[s],horizons[h])] = split_XY(Data, n_steps_in=int(histories[s]), n_steps_out=int(horizons[h]))
        
        
            d['TrainTestXY_{}_{}'.format(histories[s],horizons[h])] = (d['X_training_{}_{}'.format(histories[s],horizons[h])], d['Y_training_{}_{}'.format(histories[s],horizons[h])], d['X_testing_{}_{}'.format(histories[s],horizons[h])], d['Y_testing_{}_{}'.format(histories[s],horizons[h])], d['XNanMask_training_{}_{}'.format(histories[s],horizons[h])], d['YNanMask_training_{}_{}'.format(histories[s],horizons[h])], d['XNanMask_testing_{}_{}'.format(histories[s],horizons[h])], d['YNanMask_testing_{}_{}'.format(histories[s],horizons[h])])


   
            with open('TrainTestXY_%s_history%s_horizon%s.pkl'%(Pid_list[i],histories[s],horizons[h]), 'wb') as f:
                pickle.dump(d['TrainTestXY_{}_{}'.format(histories[s],horizons[h])], f, protocol=2)

