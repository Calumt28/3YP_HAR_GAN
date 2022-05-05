# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:03:26 2022

@author: calum
"""

import numpy as np
import collections
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import os
from sklearn import preprocessing

import skimage.io
import librosa
import librosa.display
from matplotlib import cm # colour map
import scipy
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.getcwd()
names=['labels','activities','time series','x', 'y', 'z']
activity = ['A', 'B', 'D', 'Q']
""" [walking, jogging, sitting, writing] """
frequency=20
step=200


src="C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/spectrogram/"


def multivariate_data(dataset, step):
    data = []
    start = step
    for i in range(start, len(dataset)+1, step):
        indices = range(i-step, i)
        data.append(dataset[indices])
    return np.array(data)

def create_time_steps(length):
    return list(np.arange(0, length/frequency,1/frequency))

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

def import_raw(x_start, x_split, rf): # x_split datasets, starting at x_start
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

def pre_proc_real(data_raw):
    data_full=data_raw
    data_full=data_full.loc[((data_full['activities'] == 'A') | (data_full['activities'] == 'B')
          | (data_full['activities'] == 'D') | (data_full['activities'] == 'Q'))]
 
    data_clean = pd.DataFrame(columns=names)
    
    l_1 = int(data_full['labels'].values[1])
    for i in range(40):
        
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
    #sys.exit() 
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
    scaler=preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(data) 
    
    return scaler

def pre_proc(data_raw, scaler):
    data_full=data_raw
    #data_raw=data_raw.loc[data_raw['activities']==activity]     #selected activity data

    data_full=data_full.loc[((data_full['activities'] == 'A') | (data_full['activities'] == 'B')
          | (data_full['activities'] == 'D') | (data_full['activities'] == 'Q'))]
    #print(data_full)
    data_clean = pd.DataFrame(columns=names)
    for i in range(10):
        d_sel=data_full.loc[((data_full['labels'] == str(1640+i)))]
        dataA=d_sel.loc[((d_sel['activities'] == 'A'))].head(18*200)
        dataB=d_sel.loc[((d_sel['activities'] == 'B'))].head(18*200)
        dataD=d_sel.loc[((d_sel['activities'] == 'D'))].head(18*200)
        dataQ=d_sel.loc[((d_sel['activities'] == 'Q'))].head(18*200)
        d_sel=pd.concat([dataA, dataB, dataD, dataQ], axis=0)
        #print(d_sel)
        #print("______________________")
        data_clean=pd.concat([data_clean, d_sel], axis=0)
        
    
    """
    dataA=d_sel.loc[((d_sel['activities'] == 'A'))]#.head(180*200)
    dataB=d_sel.loc[((d_sel['activities'] == 'B'))]#.head(180*200)
    dataD=d_sel.loc[((d_sel['activities'] == 'D'))]#.head(180*200)
    dataQ=d_sel.loc[((d_sel['activities'] == 'Q'))]#.head(180*200)
    
    print(dataA.shape, dataB.shape, dataD.shape, dataQ.shape)
    
    data_full=pd.concat([dataA, dataB, dataD, dataQ], axis=0)
    """
    
 
    #print(data_full)
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
    also splits activities into 4 types
    """
    
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

    # ###if L2 normalization,uncomment the code before END
    # dataset = tf.keras.utils.normalize(dataset, axis=0,order=2)#L2
    # ###END
    step=200

    x_train_single = multivariate_data(dataset, step)
    
    #code for saving plot of data for visual inspections
    #show_plot(x_train_single[1], 'Single Data')
    #np.savetxt("dataset.txt", dataset)
    #print(dataset.shape)
    #print(x_train_single.shape)
    act=36000
    #print(dataset.shape)
    #reshape to persons
    dataset=dataset.reshape((10,-1,4))
    #show_plot(dataset[5,:,1:], 'Raw_Data')
    #show_plot(dataset[0,:,1:], 'Raw_Data')
    #print(dataset.shape)
    
    #datasets is a tf dataset with a single element of shape (batch_size, step, 3)
    #datasets = tf.data.Dataset.from_tensor_slices(x_train_single)
    #randomly shuffles elements of this dataset
    #datasets = datasets.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    #print(x_train_single)
    data_n=x_train_single[:,:,1:]  
    label_n=x_train_single[:,0,0] 
    #print(collections.Counter(label_n))
    #print(label_n)
    #label_n=tf.keras.utils.to_categorical(label_n) 
        
    return dataset

def one_plot(plot_data, title, i):
    a_len = int(plot_data.shape[0]/4)
    print(plot_data.shape)
    time_steps = create_time_steps(a_len)
    plt.figure()
    plt.title(title)
    if (plot_data.ndim>1):
        plt.plot(time_steps, plot_data[a_len*i:a_len*(i+1), 0].flatten(), color='red', label='Accel-X')
        plt.plot(time_steps, plot_data[a_len*i:a_len*(i+1), 1].flatten(), color='blue', label='Accel-Y')
        plt.plot(time_steps, plot_data[a_len*i:a_len*(i+1), 2].flatten(), color='yellow', label='Accel-Z')
    else:
        plt.plot(time_steps, plot_data.flatten(), color='red', label='Accel-X')
    plt.ylim((-1, 1))
    plt.legend()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration ($m/s^2$)')
    plt.show()
    #plt.savefig(project_location+'plots/GAN/'+title+'.png')
    return plt

def show_plot(plot_data, title):
    a_len = int(plot_data.shape[0]/4)
    #time_steps = create_time_steps(a_len)
    time_steps = create_time_steps(a_len/18)
    fig, act = plt.subplots(4, sharex=True, sharey=True, figsize=(4, 8), dpi=100)
    fig.suptitle(title)
    fig.tight_layout()
    act[3].set_xlabel('Time (seconds)')
    
    for i in range(4):
        act[i].set_title(activity[i])
        if (plot_data.ndim>1):
            #act[i].plot(time_steps, plot_data[a_len*i:a_len*(i+1), 0].flatten(), color='red', label='Accel-X')
            act[i].plot(time_steps, plot_data[a_len*i:a_len*(i)+200, 0].flatten(), color='red', label='Accel-X')
            act[i].plot(time_steps, plot_data[a_len*i:a_len*(i)+200, 1].flatten(), color='blue', label='Accel-Y')
            act[i].plot(time_steps, plot_data[a_len*i:a_len*(i)+200, 2].flatten(), color='yellow', label='Accel-Z')
            act[i].set_ylabel('Acceleration ($m/s^2$)')
            act[i].set_ylim(-0.5, 1)
        else:
            print(plot_data.ndim)
            act[i].plot(time_steps, plot_data.flatten(), color='red', label='Accel-X')
        #act[i].legend()
        #act[i].set(xlabel('Time (seconds)'), ylabel('Acceleration ($m/s^2$)'))
    #plt.ax1.plt.figure()
    plt.legend()
    plt.show()
    #plt.savefig(project_location+'plots/GAN/'+title+'.png')
    return plt

def hist_plot(plot_data, title):
    a_len = int(plot_data.shape[1]/4)
    time_steps = create_time_steps(a_len)
    fig, act = plt.subplots(nrows=10, ncols=4, sharex=True, sharey=True, figsize=(16, 8), dpi=100)
    #fig.suptitle(title)
    #fig.tight_layout()
    #act[3].set_xlabel('Time (seconds)')
    #print(plot_data.shape)
    #print("size")
    #plot_data=plot_data[0]
    
    for j in range(10):
        p_data=plot_data[j]
        print(p_data.shape)
        for i in range(4):
            act[0,i].set_title(activity[i])
            if (plot_data.ndim>1):
                act[j,i].hist(p_data[a_len*i:a_len*(i+1), 0].flatten(), color='red', label='Accel-X', histtype='step', lw=1, bins=100,)
                act[j,i].hist(p_data[a_len*i:a_len*(i+1), 1].flatten(), color='blue', label='Accel-Y',histtype='step', lw=1, bins=100,)
                act[j,i].hist(p_data[a_len*i:a_len*(i+1), 2].flatten(), color='yellow', label='Accel-Z',histtype='step', lw=1, bins=100,)
                #act[i,j].set_xlabel('Acceleration ($m/s^2$)')
                act[j,i].set_ylim(0, 300)
                act[j,i].set_xlim(-0.5, 1)
            else:
                #print(plot_data.ndim)
                act[i].plot(time_steps, plot_data.flatten(), color='red', label='Accel-X')
    #plt.legend()
    plt.show()
    #plt.savefig(project_location+'plots/GAN/'+title+'.png')
    return plt

def plot_raw(real, spec, time):
    #select random from each act - {0 and 10}
    select = int(np.random.randint(10, size=1))
    #select_t = select + 20
    print(real.shape)
    print(select)
    select=7
    
    #show_plot(real[select,:,1:], str(1640+select)+'_Raw_Data_real')
    #show_plot(spec[select,:,1:], str(1640+select)+'_Raw_Data_specFake')
    #show_plot(time[select,:,1:], str(1660+select)+'_Raw_Data_timeFake')
    select=0
    #for select in range(10):
    hist_plot(real[:,:,1:], '_Hist_Data_real')
    hist_plot(spec[:,:,1:], 'Hist_Data_specFake')
    hist_plot(time[:,:,1:], str(1660+select)+'_Hist_Data_timeFake')
    
    return


mm_scaler = pre_proc_real(import_real(0, 51))
real = pre_proc(import_real(40, 10), mm_scaler)
spec = pre_proc(import_raw(40,10,'fake'), mm_scaler)
timef = pre_proc(pre_fake(), mm_scaler)


plot_raw(real,spec,timef)

#one_plot(spec[2,:,1:], "A_Raw_Data_fake", 0)
#one_plot(real[0,:,1:], "B_Raw_Data", 1)
#one_plot(real[0,:,1:], "D_Raw_Data", 2)
#one_plot(real[0,:,1:], "Q_Raw_Data", 3)



















