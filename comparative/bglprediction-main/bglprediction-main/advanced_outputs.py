
"""
Created in 2021

@author: Hoda Nemat (hoda.nemat@sheffield.ac.uk)
"""


import numpy as np
import pickle as pickle
from prediction_models import Linear_model, LSTM_multi, ConLSTM_subseq




def norm_min_max (X):
    min_BG = 40
    Max_BG = 400

    X_norm = (X - min_BG) / (Max_BG - min_BG) 
    
    return X_norm


    
def advanced_methods (PID, run, PH):


    # Loading basic models
    
    with open('Basic{}_Y_{}_{}.pkl'.format(PH, PID, str(run)), 'rb') as f:
        d = pickle.load(f) 
          

    BasicNames = ['Linear', 'VLSTM', 'BiLSTM']     
    model = {}
    history = {}
   
    # Preparing outputs for the advanced methods:
    
    subset_list = ['Test','Train']
    for s in range(len(subset_list)):
        
        d['n_steps_{}'.format(subset_list[s])] = d['yhat_{}_BiLSTM'.format(subset_list[s])].shape[1]
        d['n_samples_{}'.format(subset_list[s])] = d['yhat_{}_BiLSTM'.format(subset_list[s])].shape[0]
        
        
        # 1. stacking
        X_stacking = [d['yhat_{}_{}'.format(subset_list[s], BasicNames[b])] for b in range(len(BasicNames))]
        d['X_{}_Stacking'.format(subset_list[s])] = np.hstack(X_stacking)
       
        
        # 2. multivariate
        d['X_{}_Multivariate'.format(subset_list[s])] = np.concatenate(([d['yhat_{}_{}'.format(subset_list[s], BasicNames[b])].reshape(d['n_samples_{}'.format(subset_list[s])],d['n_steps_{}'.format(subset_list[s])],1) for b in range(len(BasicNames))]), axis=2)
        
        
        # reshape into subsequences [samples, seqs, rows, cols, channels] => [samples, 3, 1, 6, 1]        
        # 3. subseqs
        d['X_{}_Subseqs'.format(subset_list[s])] = np.concatenate(([d['yhat_{}_{}'.format(subset_list[s], BasicNames[b])].reshape(d['n_samples_{}'.format(subset_list[s])], 1, 1, d['n_steps_{}'.format(subset_list[s])], 1) for b in range(len(BasicNames))]),axis=1)
           


    # metaLearner    
    MetaNames = ['Stacking','Multivariate','Subseqs']
    MetaLearners = [Linear_model, LSTM_multi, ConLSTM_subseq]


    # Meta models
    for k in range(len(MetaLearners)):
        d['yhat_Test_{}'.format(MetaNames[k])], d['yhat_Train_{}'.format(MetaNames[k])], model['model_{}'.format(MetaNames[k])], history['history_{}'.format(MetaNames[k])] = MetaLearners[k](d['X_Train_{}'.format(MetaNames[k])], d['Y_Train'], d['X_Test_{}'.format(MetaNames[k])], d['Y_Test'], epochs=2, batch_size=32)
        
            
#        if k in [1,2]:
#            model['model_{}'.format(MetaNames[k])].save('Meta{}_Model_{}_{}_{}.h5'.format(PH, PID, MetaNames[k], run))
            

                
#    with open('Meta{}_History_{}_r{}.pkl'.format(PH, PID, run), 'wb') as f:
#        pickle.dump(history, f, protocol=2)
 
    with open('Meta{}_Y_{}_{}.pkl'.format(PH, PID, run), 'wb') as f:
        pickle.dump(d, f, protocol=2)    
        
   


def run_advance(PH):
    PID_list = ['540', '544', '552','559', '563', '567', '570', '575', '584', '588', '591', '596']           
    RUN = 5
    
    for r in range(1, RUN+1):
        for p in range(len(PID_list)):   
            advanced_methods(PID_list[p], r, PH)     
                     
                
run_advance(PH='6')        








