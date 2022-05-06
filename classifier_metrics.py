# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:03:58 2021

@author: calum

this file contains code from [1]
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten,LSTM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix 
import seaborn as sns 
import sys
import collections

from tensorflow.keras import backend as K
from tabulate import tabulate
import random

project_location = "C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/"
src="C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/spectrogram/"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.getcwd()
names=['labels','activities','time series','x', 'y', 'z']

activity = ['A', 'B', 'D', 'Q']
act_group=['Motionless','Low acceleration motion','Moderate acceleration motion','High  acceleration motion'] 

ACT_N = 4

""" WISDM data is sampled every 50ms = 20 Hz """
frequency=20
""" number of training steps """
step = 200
""" batch size per epoch """
BATCH_SIZE = 64
BUFFER_SIZE = 60000

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def import_real(x_start, x_split): # x_split datasets, starting at x_start
    x_start = x_start+1600
    data_raw = pd.read_csv(
        r"C:\Users\calum\OneDrive\Documents\Year_3_Project\Python\oldGAN\dataset\train\data_"+str(x_start)+"_accel_watch.txt"
        ,names=names, lineterminator=';', skip_blank_lines=True)
    #np.savetxt('C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/sumx.txt',data_raw,delimiter=',',fmt = '%s')

    data_raw = data_raw[:-1]
    data_raw['labels']=str(x_start)
    
    for i in range(x_start+1,x_start+x_split): #or 1651
        #print(i)
        filename='C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/dataset/train/data_'+str(i)+'_accel_watch.txt'
        x=pd.read_csv(filename,names=names, lineterminator=';')
        x['labels']=str(i)
        data_raw=pd.concat([data_raw, x[:-1]], axis=0)
    
    return data_raw


def import_inverted(x_start, x_split, rf): # x_split datasets, starting at x_start
    x_start = x_start+1600
    
    data_raw = pd.read_csv(
        r"C:\Users\calum\OneDrive\Documents\Year_3_Project\Python\spectrogram\dataset\raw\\" +str(rf)+ "\data_"+str(x_start)+"_accel_watch.txt"
        ,names=names, lineterminator=';', skip_blank_lines=True)
    #np.savetxt('C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/sumx.txt',data_raw,delimiter=',',fmt = '%s')
    data_raw['labels']=str(x_start)
    #data_raw = data_raw[]
    
    for i in range(x_start+1,x_start+x_split): #or 1651
        #print(i)
        filename=src+'/dataset/raw/'+str(rf)+'/data_'+str(i)+'_accel_watch.txt'
        x=pd.read_csv(filename,names=names, lineterminator=';')
        x['labels']=str(i)
        data_raw=pd.concat([data_raw, x], axis=0)
    
    #print(data_raw)
    return data_raw


def pre_fake():
    data_raw = pd.read_csv(
        r"C:\Users\calum\OneDrive\Documents\Year_3_Project\Python\oldGAN\dataset\gen_10\data_1660_accel_watch.txt"
        ,names=names, skip_blank_lines=True)
    #np.savetxt('C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/sumx.txt',data_raw,delimiter=',',fmt = '%s')
    data_raw['labels']=str(1660)
    
    for i in range(1661,1660+10):
        filename='C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/dataset/gen_10/data_'+str(i)+'_accel_watch.txt'
        x=pd.read_csv(filename,names=names)
        x['labels']=str(i)
        data_raw=pd.concat([data_raw, x], axis=0)
    
    return data_raw

def multivariate_data(dataset, step):
    data = []
    start = step
    for i in range(start, len(dataset)+1, step):
        indices = range(i-step, i)
        data.append(dataset[indices])
    return np.array(data)

def pre_proc_real(data_raw):
    data_full=data_raw
    data_full=data_full.loc[((data_full['activities'] == 'A') | (data_full['activities'] == 'B')
          | (data_full['activities'] == 'D') | (data_full['activities'] == 'Q'))]
 
    data_clean = pd.DataFrame(columns=names)
    
    l_1 = int(data_full['labels'].values[1])
    for i in range(50):
        
        if(i!=16):
            d_sel=data_full.loc[((data_full['labels'] == str(l_1+i)))]
           
        else:
            d_sel=data_full.loc[((data_full['labels'] == str(1650)))]
        #print(d_sel.shape)
        dataA=d_sel.loc[((d_sel['activities'] == 'A'))].head(18*200)
        dataB=d_sel.loc[((d_sel['activities'] == 'B'))].head(18*200)
        dataD=d_sel.loc[((d_sel['activities'] == 'D'))].head(18*200)
        dataQ=d_sel.loc[((d_sel['activities'] == 'Q'))].head(18*200)
        while (dataD.shape[0] < 3600):
            print(i, dataD.shape)
            new_index=dataD.index[-1]+1
            dataD=dataD.append(pd.DataFrame(index=[new_index], data=dataD.tail(1).values, columns=dataD.columns))
            #dataD = pd.concat([dataD, dataD.iloc[-1]], axis=0)
            #print(dataD)
            
        d_sel=pd.concat([dataA, dataB, dataD, dataQ], axis=0)
        #print(d_sel.shape)
        if(d_sel.shape[0]<14400):
            print(i)
            #print(dataA.shape, dataB.shape, dataD.shape, dataQ.shape)
            
        data_clean=pd.concat([data_clean, d_sel], axis=0)

    data_clean.drop(['labels', 'time series'],axis=1,inplace=True) 
    #only use 3-axis data
    #features_considered = ['x', 'y', 'z']
    #features=data_raw[features_considered]
    label=data_clean.iloc[:,0] #extract activity labels 
    #data_full=data_full[features_considered]
    data=data_clean.iloc[:,1:] #extract x,y,z data 
    s_total = data.values 
    data=pd.DataFrame(s_total) 
    
    scaler=preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(data) 
    
    return scaler


def pre_proc(data_raw, scaler):
    data_full=data_raw
    
    data_full=data_full.loc[((data_full['activities'] == 'A') | (data_full['activities'] == 'B')
          | (data_full['activities'] == 'D') | (data_full['activities'] == 'Q'))]
 
    #"clean data" - ensure all activities same size to prevent overlap when mini samples taken
    l_1 = int(data_full['labels'].values[1])
    
    data_clean = pd.DataFrame(columns=names)
    #test 10
    for i in range(10):
        d_sel=data_full.loc[((data_full['labels'] == str(l_1+i)))]
        dataA=d_sel.loc[((d_sel['activities'] == 'A'))].head(18*200)
        dataB=d_sel.loc[((d_sel['activities'] == 'B'))].head(18*200)
        dataD=d_sel.loc[((d_sel['activities'] == 'D'))].head(18*200)
        dataQ=d_sel.loc[((d_sel['activities'] == 'Q'))].head(18*200)
        d_sel=pd.concat([dataA, dataB, dataD, dataQ], axis=0)
        #print(d_sel.shape)
        data_clean=pd.concat([data_clean, d_sel], axis=0)

    #data_clean=data_full

    data_clean.drop(['labels', 'time series'],axis=1,inplace=True) 
    #only use 3-axis data
    #features_considered = ['x', 'y', 'z']
    #features=data_raw[features_considered]
    label=data_clean.iloc[:,0] #extract activity labels 
    #data_full=data_full[features_considered]
    data=data_clean.iloc[:,1:] #extract x,y,z data 
    s_total = data.values 
    data=pd.DataFrame(s_total) 
    
    """ one hot encoding to replace labels.
    """
    
    ## data type transformation 
    label=label.replace('A','0') 
    label=label.replace('B','1')
    label=label.replace('D','2')
    label=label.replace('Q','3')
    
    label=pd.DataFrame(label,dtype=int) 
    ## MinMaxScaler normalization 
    #scaler=preprocessing.MinMaxScaler(feature_range=(-1,1)) 
    #print(data.shape)
    data=scaler.transform(data)  
    data=pd.DataFrame(data, columns=['x','y','z']) 
    ## Concat label and data 
    label=label.reset_index(drop=True) 
    data=data.reset_index(drop=True) 
    dataset=pd.concat([label,data],axis=1) 
    dataset=np.array(dataset)

    step=200

    x_train_single = multivariate_data(dataset, step)
    
    #show_plot(x_train_single[0,:,1:], str(activity)+'_Min-max_Normalization')
    #show_plot(dataset[:36000,1:], 'A_Full')
    
    data_f=dataset[:,1:]  
    label_f=dataset[:,0] 
    
    data_n=x_train_single[:,:,1:]  
    label_n=x_train_single[:,0,0] 
    
    
    label_n=tf.keras.utils.to_categorical(label_n) 
    label_f=tf.keras.utils.to_categorical(label_f) 
    
    return data_n, label_n

def show_plot(plot_data, title):
    time_steps = create_time_steps(plot_data.shape[0])
    plt.figure()
    plt.title(title)
    plt.plot(time_steps, plot_data[:, 0].flatten(), color='red', label='Accel-X')
    plt.plot(time_steps, plot_data[:, 1].flatten(), color='blue', label='Accel-Y')
    plt.plot(time_steps, plot_data[:, 2].flatten(), color='yellow', label='Accel-Z')
    plt.legend()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration ($m/s^2$)')
    plt.show()
    plt.savefig(project_location+'plots/GAN/'+title+'.png')
    return plt   

def create_time_steps(length):
    return list(np.arange(0, length/frequency,1/frequency))

def show_plot(plot_data, title):
    time_steps = create_time_steps(plot_data.shape[0])
    plt.figure()
    plt.title(title)
    plt.plot(time_steps, plot_data[:, 0].flatten(), color='red', label='Accel-X')
    plt.plot(time_steps, plot_data[:, 1].flatten(), color='blue', label='Accel-Y')
    plt.plot(time_steps, plot_data[:, 2].flatten(), color='yellow', label='Accel-Z')
    plt.legend()
    plt
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration ($m/s^2$)')
    plt.show()
    #plt.savefig(project_location+'plots/GAN/'+title+'.png')
    return plt

def test_model(model, d_value, d_label, real):
    label_test=d_label
    s=558
    show_plot(d_value.reshape(-1, 3), "1647_Q_Min_max")

    data_test, label_test = d_value,d_label
    #print(d_label.shape, d_value.shape)
    #sys.exit()
    rng_state = np.random.get_state()
    np.random.shuffle(data_test)
    np.random.set_state(rng_state)
    np.random.shuffle(label_test)
    
    #test model on test dataset
    predictions=model.predict(data_test) 
    
    #get array of test labels
    max_true = np.argmax(label_test, axis = 1)

    max_prediction = np.argmax(predictions, axis = 1)
    
    #total is number of entries in test dataset
    correct = 0
    skip = 0
    total = max_prediction.shape[0]  
    #print(total)
    
    #set all activity accuracy to 0 by default
    #[correct_n, incorrect_n]
    
    activity_accuracy = np.zeros((ACT_N, 2))
    
    #loop through set of predictions, comparing predicted label to true label
    for i in range(total):
        if max_true[i] >= 4:
            #skip
            skip = skip + 1
            continue
        #print(max_true[i])
        if max_true[i] == max_prediction[i]:
            correct = correct + 1
            #increment correct for this activity
            activity_accuracy[max_true[i], 0] = activity_accuracy[max_true[i], 0] + 1
        else:
            #increment incorrect
            activity_accuracy[max_true[i], 1] = activity_accuracy[max_true[i], 1] + 1
    
    total_accuracy = (correct/(total-skip))
    
    #print(str(correct)+'/'+str(total)+' = '+str(round(100*correct/total, 2))+'%')
    
    #set accuracy based off correct/total
    for i in range(ACT_N):
        activity_accuracy[i, 0] = activity_accuracy[i, 0] / (activity_accuracy[i, 0]+activity_accuracy[i, 1])   
    #plot CM   
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

    mpl.rc('font', **font)

    CM = confusion_matrix(max_true, max_prediction) 
    plt.figure(figsize=(16,14)) 
    sns.heatmap(CM, xticklabels = activity, yticklabels = activity, 
    annot = True, fmt = 'd',cmap='Blues') 
    plt.title('Confusion Matrix') 
    plt.xlabel('Predicted Label') 
    plt.ylabel('True Label') 
    plt.show()  
    
    #return table of activity accuracy + total accuracy
    return np.append(activity_accuracy[:,0], total_accuracy)


def metrics():
    """ load models """
    r_model=keras.models.load_model(project_location+'models/cnn_opt/CNN40RR_clean2', custom_objects={'f1_m':f1_m, 'recall_m':recall_m, 'precision_m':precision_m})
    #r_model=keras.models.load_model(project_location+'models/CNN40RRfv', custom_objects={'f1_m':f1_m, 'recall_m':recall_m, 'precision_m':precision_m})
    #r_model=keras.models.load_model(project_location+'models/CNN40RR', custom_objects={'f1_m':f1_m, 'recall_m':recall_m, 'precision_m':precision_m})
    #f_model=keras.models.load_model(project_location+'models/CNN40FR', custom_objects={'f1_m':f1_m, 'recall_m':recall_m, 'precision_m':precision_m})
    
    
    """ load datasets """
    
    min_max_scaler=pre_proc_real(import_real(0, 51))

    real, real_label = pre_proc(import_real(40, 10),min_max_scaler)

    real_inv, real_inv_label = pre_proc(import_inverted(40, 10, "real_inv"),min_max_scaler)

    r1 = np.random.normal(0, 0.2, (720, 200, 3))
    
    spec_fake, spec_fake_label = pre_proc(import_inverted(40, 10, "fake"),min_max_scaler)

    time_fake, time_fake_label = pre_proc(pre_fake(),min_max_scaler)


    """ compare predictions to true labels with test"""
    act_acc_real = test_model(r_model, real, real_label, False)
    act_acc_rInv = test_model(r_model, real_inv, real_inv_label, False)
    act_acc_spec = test_model(r_model, spec_fake, spec_fake_label, False)
    act_acc_time = test_model(r_model, time_fake, time_fake_label, False)
    
    
    """ printout table """
    act_type = ['Motionless','Low acceleration motion','Moderate acceleration motion motion','High  acceleration motion'] 
    #act=
    a_a = np.concatenate((np.append(activity, 'Total Motion')
                          , np.round(act_acc_real, 4)
                          , np.round(act_acc_rInv, 4)
                          , np.round(act_acc_spec, 4)
                          , np.round(act_acc_time, 4)
                          ))
    a_a = np.reshape(a_a, (ACT_N+1, 5), order='F')
    
    print(tabulate(a_a, headers = ['Activity', 'TRTR Accuracy', 'TRTR Invert', 'TRTF Spec', 'TRTF TimeSeries']))
        
    return


metrics()
