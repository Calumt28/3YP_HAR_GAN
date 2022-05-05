# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:11:43 2022

@author: calum
"""

import numpy as np
import collections
import pandas as pd
import scipy
import tensorflow
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn import preprocessing, neighbors
from sklearn.datasets import make_blobs
from sklearn.model_selection import LeaveOneOut
#from sklearn.model_selection import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics import hamming_loss
import skimage.io
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

src="C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/spectrogram/"

#names=['x', 'y', 'z']
project_location = "C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/"
data_in=project_location+'dataset/train/spectrogram/'
activity = ['A', 'B', 'D', 'Q']
names=['labels','activities','time series','x', 'y', 'z']

spec_in='C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/spectrogram/'


def import_real(x_start, x_split): # x_split datasets, starting at x_start
    x_start = x_start+1600
    data_raw = pd.read_csv(
        r"C:\Users\calum\OneDrive\Documents\Year_3_Project\Python\spectrogram\dataset\raw\real\data_"+str(x_start)+"_accel_watch.txt"
        ,names=names, lineterminator=';', skip_blank_lines=True)

    data_raw = data_raw[:-1]
    data_raw['labels']=str(x_start)
    
    for i in range(x_start+1,x_start+x_split): #or 1651
        #print(i)
        filename=src+'/dataset/raw/real/data_'+str(i)+'_accel_watch.txt'
        x=pd.read_csv(filename,names=names, lineterminator=';')
        x['labels']=str(i)
        data_raw=pd.concat([data_raw, x[:-1]], axis=0)
    
    return data_raw

def import_inverted(x_start, x_split, rf): # x_split datasets, starting at x_start
    x_start = x_start+1600
        
    data_raw = pd.read_csv(
        r"C:\Users\calum\OneDrive\Documents\Year_3_Project\Python\spectrogram\dataset\raw\\" +str(rf)+ "\data_"+str(x_start)+"_accel_watch.txt"
        ,names=names, lineterminator=';', skip_blank_lines=True)
    data_raw['labels']=str(x_start)
        
    for i in range(x_start+1,x_start+x_split): #or 1651
            #print(i)
            filename=src+'/dataset/raw/'+str(rf)+'/data_'+str(i)+'_accel_watch.txt'
            x=pd.read_csv(filename,names=names, lineterminator=';')
            x['labels']=str(i)
            data_raw=pd.concat([data_raw, x], axis=0)
        
        #print(data_raw)
    return data_raw

def import_spec():#10     
    data_raw = import_inverted(40, 10, "fake")
    return data_raw

def pre_fake():
    data_raw = pd.read_csv(
        r"C:\Users\calum\OneDrive\Documents\Year_3_Project\Python\spectrogram\dataset\raw\old_fake\data_1660_accel_watch.txt"
        ,names=names, skip_blank_lines=True)
    data_raw['labels']=str(1640)
    
    for i in range(1661,1660+10):
        filename=src+'/dataset/raw/old_fake/data_'+str(i)+'_accel_watch.txt'
        x=pd.read_csv(filename,names=names)
        x['labels']=str(i-20)
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

def pre_proc(data_raw, scaler, count=10):
    data_full=data_raw
    #data_raw=data_raw.loc[data_raw['activities']==activity]     #selected activity data

    data_full=data_full.loc[((data_full['activities'] == 'A') | (data_full['activities'] == 'B')
          | (data_full['activities'] == 'D') | (data_full['activities'] == 'Q'))]
 
    data_clean = pd.DataFrame(columns=names)
    
    l_1 = int(data_full['labels'].values[1])
    for i in range(count):
        
        if((l_1+i)!=1616):
            d_sel=data_full.loc[((data_full['labels'] == str(l_1+i)))]
           
        else:
            d_sel=data_full.loc[((data_full['labels'] == str(l_1)))]
        dataA=d_sel.loc[((d_sel['activities'] == 'A'))].head(18*200)
        dataB=d_sel.loc[((d_sel['activities'] == 'B'))].head(18*200)
        dataD=d_sel.loc[((d_sel['activities'] == 'D'))].head(18*200)
        dataQ=d_sel.loc[((d_sel['activities'] == 'Q'))].head(18*200)
        while (dataD.shape[0] < 3600):
            new_index=dataD.index[-1]+1
            dataD=dataD.append(pd.DataFrame(index=[new_index], data=dataD.tail(1).values, columns=dataD.columns))
        d_sel=pd.concat([dataA, dataB, dataD, dataQ], axis=0)
        #print(d_sel.shape)
        if(d_sel.shape[0]<14400):
            print(i)
            #print(dataA.shape, dataB.shape, dataD.shape, dataQ.shape)
            
        data_clean=pd.concat([data_clean, d_sel], axis=0)
    data_clean.drop(['labels', 'time series'],axis=1,inplace=True) 
    #only use 3-axis data
    label=data_clean.iloc[:,0] #extract activity labels 
    #data_full=data_full[features_considered]
    data=data_clean.iloc[:,1:] #extract x,y,z data 
    s_total = data.values 
    data=pd.DataFrame(s_total) 
    
    ## data type transformation 
    label=label.replace('A','0') 
    label=label.replace('B','1')
    label=label.replace('D','2')
    label=label.replace('Q','3')

    label=pd.DataFrame(label,dtype=int) 
    ## MinMaxScaler normalization 
    data=scaler.transform(data)  
    data=pd.DataFrame(data, columns=['x','y','z']) 
    ## Concat label and data 
    label=label.reset_index(drop=True) 
    data=data.reset_index(drop=True) 
    dataset=pd.concat([label,data],axis=1) 
    dataset=np.array(dataset)

    step=200

    x_train_single = multivariate_data(dataset, step)
    
    data_n=x_train_single[:,:,1:]  
    label_n=x_train_single[:,0,0] 
    
    print(collections.Counter(label_n))
    #print(label_n)
    label_n=tf.keras.utils.to_categorical(label_n) 
        
    return data_n

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

def print_layers(model, model_in):
    table=pd.DataFrame(columns=["Name","Type","Shape"])
    table = table.append({"Name":model_in, "Type": "InputLayer","Shape":model.input_shape}, ignore_index=True)
    for layer in model.layers:
        table = table.append({"Name":layer.name, "Type": layer.__class__.__name__,"Shape":layer.output_shape}, ignore_index=True)
    print(model.name)
    print("_________________________________________________________________")
    print(table)
    print("_________________________________________________________________")
    return

""" create feature extraction model """
cnn_model = keras.models.load_model(project_location+'models/cnn_opt/CNN40RR_clean2', custom_objects={'f1_m':f1_m, 'recall_m':recall_m, 'precision_m':precision_m})

cnn_model.trainable=False

cnn_model.pop()
cnn_model.pop()
print_layers(cnn_model, "x")
#cnn_model.summary()

from sklearn.metrics import accuracy_score


def LOO_NN(x, y):
    # enumerate splits
    y_true, y_pred = list(), list()
    cv = LeaveOneOut()
    #print(X)
    for train_ix, test_ix in cv.split(x):
    	# split data
        #print(X.take(train_ix, axis=0).shape)
        #print(train_ix, test_ix)
        #x_train, x_test = x.take(train_ix, axis=0).take(train_ix, axis=1), x.take(test_ix, axis=0).take(train_ix, axis=1)
        x_train, x_test = x.take(train_ix, axis=0), x.take(test_ix, axis=0)
        y_train, y_test = y[train_ix], y[test_ix]
    	# fit model
        #print(x_test.shape)
        model = neighbors.KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        model.fit(x_train, y_train)
    	# evaluate model
        yhat = model.predict(x_test)
    	# store
        #if(y_test == 1):
        y_true.append(y_test[0])
        y_pred.append(yhat[0])
        
    # calculate accuracy
    a_len2 = int(np.array(y_true).shape[0]/2)
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    #print(np.sum(y_pred), np.sum(y_true))
    acc_r = accuracy_score(y_true[:a_len2], y_pred[:a_len2])
    acc_f = accuracy_score(y_true[a_len2:], y_pred[a_len2:])
    #print(y_pred)
    #print('Accuracy Real: %.3f' % acc_r)
    #print('Accuracy Fake: %.3f' % acc_f)
    return acc_r, acc_f

MMscaler=pre_proc_real(import_real(0, 51))

#real = pre_proc(import_real_test(), MMscaler)
real2 = pre_proc(import_real(40, 10), MMscaler, count=10)
real3 = pre_proc(import_real(0, 10), MMscaler, count=10)
#real3 = pre_proc(import_inverted(30, 10, 'real'), MMscaler, count=10)
#print(real.shape, real2.shape)
#sys.exit()

fake_spec = pre_proc(import_spec(), MMscaler)
fake_time = pre_proc(pre_fake(), MMscaler)

#a_len = int(real.shape[0]/4)
a_len = int(real2.shape[0]/4)
#a_len = int(real.shape[0]*real.shape[1]/4)


acc_spec = np.zeros((4,2))
acc_time = np.zeros((4,2))
acc_real = np.zeros((4,2))

frequency=20
def create_time_steps(length):
    return list(np.arange(0, length/frequency,1/frequency))


def one_plot(plot_data, title):
    a_len = int(plot_data.shape[0]/2)
    #print(plot_data.shape)
    #print(a_len)
    time_steps = create_time_steps(plot_data.shape[1]*a_len)
    #print(time_steps)
    plt.figure()
    plt.title(title)
    if (plot_data.ndim>1):
        plt.hist(plot_data[0:a_len,:].flatten(), color='red', label='Real', histtype='step', bins=1000)
        plt.hist(plot_data[a_len:, :].flatten(), color='blue', label='Fake', histtype='step', bins=1000)
    else:
        plt.plot(time_steps, plot_data.flatten(), color='red', label='Accel-X')
        print("hi")
    plt.xlim((-0.5, 0.5))
    plt.legend()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration ($m/s^2$)')
    plt.show()
    #plt.savefig(project_location+'plots/GAN/'+title+'.png')
    return plt
#print(acc_spec.shape)

for i in range(4):
    xr = real2[a_len*i:a_len*(i+1),:,:]
    np.random.shuffle(xr)
    xf = fake_spec[a_len*i:a_len*(i+1),:,:]
    np.random.shuffle(xf)
    X = (np.vstack((xr, xf)))
    x = np.array(cnn_model(X))
    x_len = int(x.shape[0]/2)
    
    label = np.concatenate((np.ones((x_len)), np.zeros((x_len))))
    y = (label.ravel())
    
    acc_spec[i,0], acc_spec[i,1] = LOO_NN(x, y)
    print(acc_spec[i,0], acc_spec[i,1])

print("____________")


for i in range(4):
    #X = (np.vstack((real2[a_len*i:a_len*(i+1),:,:], fake_time[a_len*i:a_len*(i+1),:,:])))
    xr = real2[a_len*i:a_len*(i+1),:,:]
    np.random.shuffle(xr)
    xf = fake_time[a_len*i:a_len*(i+1),:,:]
    np.random.shuffle(xf)
    X = (np.vstack((xr, xf)))
    x = np.array(cnn_model(X))
    x_len = int(x.shape[0]/2)
    
    
    label = np.concatenate((np.ones((x_len)), np.zeros((x_len))))
    y = (label.ravel())
    
    acc_time[i,0], acc_time[i,1] = LOO_NN(x, y)
    print(acc_time[i,0], acc_time[i,1])
    
print("____________")



r1 = np.random.normal(0, 0.1, (180, 200, 3))
r2 = np.random.normal(0, 0.2, (180, 200, 3))
"""
#print(real2.shape, real3.shape)
for i in range(4):
    X = (np.vstack((real2[a_len*i:a_len*(i+1),:,:], fake_spec[a_len*i:a_len*(i+1),:,:])))
    #X = (np.vstack((real2[a_len*i:a_len*(i+1),:,:], r1)))
    #print(X.shape)
    #sys.exit()
    #r1 = np.random.normal(0, 0.2, (180, 200, 3))
    #r2 = np.random.normal(0, 0.2, (180, 200, 3))
    #X = np.vstack((r1, r2))
    #print(X)
    #np.random.shuffle(X)
    x = np.array(cnn_model(X))
    #rng_state = np.random.get_state()
    #np.random.shuffle(X)
    #np.random.set_state(rng_state)
    #np.random.shuffle(label_train)
    #x = X.reshape((360,-1))

    #x = np.vstack((r1, r2))
    #print(x)
    x_len = int(x.shape[0]/2)
    
    x_real = x[0:x_len,:]
    x_fake = x[x_len:, :]
    #print(x_fake)
    #sys.exit()
    
    label = np.concatenate((np.ones((x_len)), np.zeros((x_len))))
    y = (label.ravel())
    #print(y)
    
    #x_real = x[0:x_len,:]
    #x_fake = x[x_len:,:]
    
    acc_real[i,0], acc_real[i,1] = LOO_NN(x, y)
    
    #acc_time[i,0], acc_time[i,1] = knns.acc_real, knns.acc_fake
    print(acc_real[i,0], acc_real[i,1])
    #one_plot(x, "feature bins")
    #one_plot(X, "raw bins")
    #sys.exit()
"""
print("____________")
print("____________")
#print(np.mean(acc_spec[:,0]), np.mean(acc_spec[:,1]))
print("____________")
#print(np.mean(acc_time[:,0]), np.mean(acc_time[:,1]))
print("____________")
#print(np.mean(acc_real[:,0]), np.mean(acc_real[:,1]))
