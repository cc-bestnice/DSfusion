
"""
Created in 2021

@author: Hoda Nemat (hoda.nemat@sheffield.ac.uk)
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Bidirectional, ConvLSTM2D, Flatten
from sklearn.linear_model import LinearRegression
from keras.callbacks import ReduceLROnPlateau





def Baseline_model(X_Train, Y_Train, X_Test, Y_Test, epochs=500, batch_size=32):
    
    # Converting notmalised X to normal
    min_BG = 40
    Max_BG = 400
    
    X_Test_real = (X_Test*(Max_BG - min_BG)) + min_BG
    X_Train_real = (X_Train*(Max_BG - min_BG)) + min_BG

    
    yhat = np.zeros(shape=(Y_Test.shape[0], Y_Test.shape[1]))
    yhat_Train = np.zeros(shape=(Y_Train.shape[0], Y_Train.shape[1]))
    

    for i in range(len(X_Test_real)):
        yhat[i] = X_Test_real[i,-1]

        
    for j in range(len(X_Train_real)):
        yhat_Train[j] = X_Train_real[j,-1]
        
    model = 0
    history = 0    

    return   yhat, yhat_Train, model, history



def LSTM_model(X_Train, Y_Train, X_Test, Y_Test, epochs=500, batch_size=32):
    
    n_timesteps = X_Train.shape[1]
    n_features = 1
    X_Train = X_Train.reshape(X_Train.shape[0], n_timesteps, n_features)
    
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(Y_Train.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    earlystop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_delta=0.0001, verbose=1)
    callbacks_list = [earlystop]
    
    history=model.fit(X_Train, Y_Train, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.20, shuffle=False, verbose=0)    

    
    yhat_Train = model.predict(X_Train, verbose=0)
    
    min_BG = 40
    Max_BG = 400

    yhat_Train = np.where(yhat_Train > Max_BG, Max_BG, yhat_Train)
    yhat_Train = np.where(yhat_Train < min_BG, min_BG, yhat_Train)

    X_Test = X_Test.reshape(X_Test.shape[0], X_Test.shape[1], n_features)
    yhat = model.predict(X_Test, verbose=0)

    yhat = np.where(yhat > Max_BG, Max_BG, yhat)
    yhat = np.where(yhat < min_BG, min_BG, yhat)
    

    
    return yhat, yhat_Train, model, history




def LSTMBi_model(X_Train, Y_Train, X_Test, Y_Test, epochs=500, batch_size=32):
    
    n_timesteps = X_Train.shape[1]
    n_features = 1
    X_Train = X_Train.reshape(X_Train.shape[0], n_timesteps, n_features)
    
    model = Sequential()
    model.add(Bidirectional(LSTM(200, activation='relu'), input_shape=(n_timesteps, n_features)))
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(Y_Train.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    earlystop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_delta=0.0001, verbose=1)
    callbacks_list = [earlystop]
    
    history=model.fit(X_Train, Y_Train, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.20, shuffle=False, verbose=0)    
    
    yhat_Train = model.predict(X_Train, verbose=0)


    min_BG = 40
    Max_BG = 400
    yhat_Train = np.where(yhat_Train > Max_BG, Max_BG, yhat_Train)
    yhat_Train = np.where(yhat_Train < min_BG, min_BG, yhat_Train)
    
    
    X_Test = X_Test.reshape(X_Test.shape[0], n_timesteps, n_features)
    yhat = model.predict(X_Test, verbose=0)
    yhat = np.where(yhat > Max_BG, Max_BG, yhat)
    yhat = np.where(yhat < min_BG, min_BG, yhat)
    
    return yhat, yhat_Train, model, history






def Linear_model (X_Train, Y_Train, X_Test, Y_Test, epochs=500, batch_size=32):
    model = LinearRegression()
    model.fit(X_Train, Y_Train)

    yhat_Train = model.predict(X_Train)
    
    min_BG = 40
    Max_BG = 400
    yhat_Train = np.where(yhat_Train > Max_BG, Max_BG, yhat_Train)
    yhat_Train = np.where(yhat_Train < min_BG, min_BG, yhat_Train)
    
    
    yhat = model.predict(X_Test)
    yhat = np.where(yhat > Max_BG, Max_BG, yhat)
    yhat = np.where(yhat < min_BG, min_BG, yhat)
    history=0
    return yhat, yhat_Train, model, history
   


# The shape of the one sample with three time steps and two variables must be [1, 3, 2]

def LSTM_multi(X_Train, Y_Train, X_Test, Y_Test, epochs=500, batch_size=32):
    
    n_timesteps = X_Train.shape[1]
    n_features = X_Train.shape[2]

    
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(Y_Train.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    earlystop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_delta=0.0001, verbose=1)
    callbacks_list = [earlystop]
    
    history=model.fit(X_Train, Y_Train, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.20, shuffle=False, verbose=0)       
    
    yhat_Train = model.predict(X_Train, verbose=0)

    min_BG = 40
    Max_BG = 400
    yhat_Train = np.where(yhat_Train > Max_BG, Max_BG, yhat_Train)
    yhat_Train = np.where(yhat_Train < min_BG, min_BG, yhat_Train)
    
    
    yhat = model.predict(X_Test, verbose=0)
    yhat = np.where(yhat > Max_BG, Max_BG, yhat)
    yhat = np.where(yhat < min_BG, min_BG, yhat)
    
    
    return yhat, yhat_Train, model,history



def ConLSTM_subseq(X_Train, Y_Train, X_Test, Y_Test, epochs=500, batch_size=32):
    
    n_samples = X_Train.shape[0]
    n_steps= X_Train.shape[1] #number of subsequences
    n_length = X_Train.shape[3]
    n_features = 1
    n_outputs = Y_Train.shape[1]
    

    
    # reshape output into [samples, timesteps, features]
    Y_Train = Y_Train.reshape(n_samples, Y_Train.shape[1], 1)
    


    model = Sequential()
    model.add(ConvLSTM2D(64, (1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
    model.add(Flatten())
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')

    
    earlystop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_delta=0.0001, verbose=1)
    callbacks_list = [earlystop]
    
    history=model.fit(X_Train, Y_Train, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.20, shuffle=False, verbose=0) 
    
    yhat_Train = model.predict(X_Train, verbose=0)
    
    
    min_BG = 40
    Max_BG = 400
    yhat_Train = np.where(yhat_Train > Max_BG, Max_BG, yhat_Train)
    yhat_Train = np.where(yhat_Train < min_BG, min_BG, yhat_Train)
    
    yhat_Train = yhat_Train.reshape(yhat_Train.shape[0], yhat_Train.shape[1])


    yhat = model.predict(X_Test, verbose=0)
    yhat = np.where(yhat > Max_BG, Max_BG, yhat)
    yhat = np.where(yhat < min_BG, min_BG, yhat)
    
    yhat = yhat.reshape(yhat.shape[0], yhat.shape[1])
    
    
    return yhat, yhat_Train, model, history








