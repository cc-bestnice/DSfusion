
"""
Created in 2021

@author: Hoda Nemat (hoda.nemat@sheffield.ac.uk)
"""

import pickle as pickle
from prediction_models import Baseline_model, Linear_model, LSTM_model, LSTMBi_model


def norm_min_max (X):
    min_BG = 40
    Max_BG = 400

    X_norm = (X - min_BG) / (Max_BG - min_BG) 
    
    return X_norm



def load_input(PID, model, PH):

    
    if model == 'VLSTM':
        history = '18'
        
    elif model == 'BiLSTM':
        history = '12'
        
    else:
        history = '6'
        
        
    with open('TrainTestXY_'+ PID +'_history{}_horizon{}.pkl'.format(history, PH), 'rb') as f:
        X_Train, Y_Train, X_Test, Y_Test, _, _, _, _ = pickle.load(f)
        
    X_Train = norm_min_max(X_Train)
    X_Test = norm_min_max(X_Test) 
    
    return X_Train, Y_Train, X_Test, Y_Test
    
def basic_methods (PID, run, PH):
    
    

    d = {}
    model = {}
    history = {}

    
    BasicModels = [Baseline_model, Linear_model, LSTM_model, LSTMBi_model]
    BasicNames = ['Baseline', 'Linear', 'VLSTM', 'BiLSTM']   

     # Basic models
    for k in range(len(BasicModels)):
        
        d['X_Train_{}'.format(BasicNames[k])], d['Y_Train_{}'.format(BasicNames[k])], d['X_Test_{}'.format(BasicNames[k])], d['Y_Test_{}'.format(BasicNames[k])] = load_input(PID, BasicNames[k], PH)
        d['yhat_Test_{}'.format(BasicNames[k])], d['yhat_Train_{}'.format(BasicNames[k])], model['model_{}'.format(BasicNames[k])], history['history_{}'.format(BasicNames[k])] = BasicModels[k]( d['X_Train_{}'.format(BasicNames[k])], d['Y_Train_{}'.format(BasicNames[k])], d['X_Test_{}'.format(BasicNames[k])], d['Y_Test_{}'.format(BasicNames[k])], epochs=2, batch_size=32)
        
            
#        if k in [2,3]:
#            model['model_{}'.format(BasicNames[k])].save('Basic{}_Model_{}_{}_{}.h5'.format(PH, PID, BasicNames[k], run))
            

            
            
#    with open('Basic{}_History_{}_{}.pkl'.format(PH, PID, run), 'wb') as f:
#        pickle.dump(history, f, protocol=2)
 
    with open('Basic{}_Y_{}_{}.pkl'.format(PH, PID, run), 'wb') as f:
        pickle.dump(d, f, protocol=2)    
            
   
    return d
 
def run_basic(PH):  
    PID_list = ['540', '544', '552', '559', '563', '567', '570', '575', '584', '588', '591', '596']           
    RUN = 5
    
    for r in range(1, RUN+1):
        for p in range(len(PID_list)):
            
            basic_methods(PID_list[p], r, PH)     
                     

run_basic(PH='6')






