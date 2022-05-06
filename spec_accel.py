# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:38:18 2022

@author: calum

this file contains code from [1]
"""

#### the code for GAN####
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

import skimage.io
import librosa
import librosa.display
from matplotlib import cm # colour map
import scipy
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.getcwd()
names=['labels','activities','time series','x', 'y', 'z']
frequency=20
step = 200
#BATCH_SIZE = 64
""" paper says batch size is 256 for speed """
BATCH_SIZE = 256
BUFFER_SIZE = 60000
#pd.read_csv(r"C:\Users\abc\Desktop\tempfi\watch\accel_available\data_1600_accel_watch.txt",names=names)

project_location = "C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/"
#proj_location2 = "C:\Users\calum\OneDrive\Documents\Year_3_Project\Python\oldGAN\"

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def import_data():
    
    
    sumx = pd.read_csv(
        r"C:\Users\calum\OneDrive\Documents\Year_3_Project\Python\oldGAN\dataset\train\data_1600_accel_watch.txt"
        #r"C:\Users\calum\OneDrive\Documents\Year_3_Project\Python\oldGAN\dataset\train\data_1600_accel_watch.txt"
        ,names=names, lineterminator=';', skip_blank_lines=True)
    #np.savetxt('C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/sumx.txt',sumx,delimiter=',',fmt = '%s')

#activity='C' #which activity to generate

    sumx = sumx[:-1]
    sumx['labels']=str(1600)
    
    for i in range(1601,1600+50): #or 1651
        filename='C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/dataset/train/data_'+str(i)+'_accel_watch.txt'
        # filename='C:/Users/abc/Desktop/tempfi/phoneacc/data_'+str(i)+'_accel_phone.txt'
        x=pd.read_csv(filename,names=names, lineterminator=';')
        
        x['labels']=str(i)
        
        #print(x)
        sumx=pd.concat([sumx, x[:-1]], axis=0)
        
        
    #testing dataset - 10 fake
    
    #data_raw = pd.read_csv(
    #    r"C:\Users\calum\OneDrive\Documents\Year_3_Project\Python\oldGAN\dataset\gen_10\data_1660_accel_watch.txt"
    #    ,names=names, skip_blank_lines=True)
    
    for i in range(1661,1661):
        filename='C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/dataset/gen_10/data_'+str(i)+'_accel_watch.txt'
        #x=pd.read_csv(filename,names=names)
        #data_raw=pd.concat([data_raw, x], axis=0)
        
    #print(sumx.shape[0])
    
    return sumx


def griff_lim(S, hop, nfft):
    return librosa.griffinlim(S, hop_length=hop, n_fft=nfft, n_iter=16, momentum=0.8, length=3600, init=None)


def pre_proc(sumx, activity, label_id):
    SUMX=sumx                                       #SUMX is all data
    sumx=sumx.loc[sumx['activities']==activity]     #sumx is selected activity data

    #only use 3-axis data
    features_considered = ['x', 'y', 'z']
    features=sumx[features_considered]
    SUMX=SUMX[features_considered]

    dataset = features.values
    datasetSUMX=SUMX.values

    dataset_raw=dataset
    #print(dataset.shape)
    
    if (dataset.shape[0] == 0):
        return 0, 0 
    
    if (dataset.shape[0] < 3600):
        print(label_id)
    
    while (dataset.shape[0] > 3600):
        #print(dataset.shape)
        dataset = dataset[:-1]
        
    while (dataset.shape[0] < 3600):
        
        dataset = np.vstack((dataset, dataset[-1,:] ))
        print(dataset.shape)
        
    #show_plot(dataset[:,0], str(activity)+'_Raw')

    ## MinMaxScaler normalization
    min_max_scaler=preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0), copy=True).fit(SUMX)
    #dataset=min_max_scaler.transform(dataset)
    
    
    x_train_single = multivariate_data(dataset, step)
    x_train_single_raw=multivariate_data(dataset_raw, step)
    
    nfft=256
    hop=int(nfft/4)#28
    hop=28
    #plt.style.use('light_background')
    #print("TESTSTESt")
   
    
    #fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True)
    Dx = librosa.stft(dataset[:,0], hop_length=hop, n_fft=nfft, window=scipy.signal.windows.hann)  # STFT of x
    Dy = librosa.stft(dataset[:,1], hop_length=hop, n_fft=nfft, window=scipy.signal.windows.hann)  # STFT of x
    Dz = librosa.stft(dataset[:,2], hop_length=hop, n_fft=nfft, window=scipy.signal.windows.hann)  # STFT of x
    
    print(Dx.shape)
    #sys.exit()
    """ drop phase of signal - keep amplitude"""
    height = Dx.shape[0]
    width = Dx.shape[1]
    
    S_mag = np.empty((height, width, 3))
    S_mag[:,:,0] = np.abs(Dx)
    S_mag[:,:,1] = np.abs(Dy)
    S_mag[:,:,2] = np.abs(Dz)
    
    S_db = np.empty((height, width, 3))
    S_db[:,:,0] = librosa.amplitude_to_db(np.abs(Dx), ref=1700)
    S_db[:,:,1] = librosa.amplitude_to_db(np.abs(Dy), ref=1700)
    S_db[:,:,2] = librosa.amplitude_to_db(np.abs(Dz), ref=1700)
    
    
    if((np.max(np.abs(Dx))>=1700 or np.max(np.abs(Dy))>=1700 or np.max(np.abs(Dz))>=1700)):
        print(np.max(np.abs(Dx)), np.max(np.abs(Dy)), np.max(np.abs(Dz)))

    S_db=S_db.reshape((height*width, 3))
    img_scaler=preprocessing.MinMaxScaler(feature_range=(0, 255), copy=True).fit(S_db)
    img=img_scaler.transform(S_db)
    
    img=img.reshape(height, width, 3)
    # min-max scale to fit inside 8-bit range
    #img = scale_minmax(S_db, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy
    #save spectorgram images
    out = project_location+'dataset/train/spectrogram/test_gan/Activity'+str(activity)+'/'+str(label_id)+'.png'
    img = img.astype(np.uint8)
    skimage.io.imsave(out, img)
    
    """ invert """
    img = 255-img
    img = np.flip(img, axis=0)
    img=img.reshape(height*width, 3)
    S_db = img_scaler.inverse_transform(img).reshape((height, width, 3))
    #S_db = img.reshape((129, 129, 3))
    
    Dx = librosa.db_to_amplitude(S_db[:,:,0], ref=1700)
    Dy = librosa.db_to_amplitude(S_db[:,:,1], ref=1700)
    Dz = librosa.db_to_amplitude(S_db[:,:,2], ref=1700)
    
    
    #invx = griff_lim(np.abs(Dy), nfft, hop)
    invx = librosa.griffinlim(Dx, hop_length=hop, n_fft=nfft, n_iter=16, momentum=1, length=3600, init=None, window=scipy.signal.windows.hann) 
    invy = librosa.griffinlim(Dy, hop_length=hop, n_fft=nfft, n_iter=16, momentum=1, length=3600, init=None, window=scipy.signal.windows.hann) 
    invz = librosa.griffinlim(Dz, hop_length=hop, n_fft=nfft, n_iter=16, momentum=1, length=3600, init=None, window=scipy.signal.windows.hann) 
    
    inv = np.empty((3600, 3))
    inv[:,0] = invx
    inv[:,1] = invy
    inv[:,2] = invz
    
    x_train_inv = multivariate_data(inv, step)
    
    #show_plot(x_train_single[0,:,0], str(activity)+'_Raw_Min-max_')
    show_plot(x_train_single[0], str(activity)+'_Raw_Min-max_')
    show_plot(x_train_inv[0], str(activity)+'_inv_Min-max_')
    show_plot(dataset, str(activity)+'_Raw_Full_')
    show_plot(inv, str(activity)+'_Inv_Full_')
    
    
    #f = project_location+'dataset/train/spectrogram/GriffinLim/real/no_label/Activity'+str(activity)+'/data_'+str(label_id)+'_accel_watch.txt'
    f = project_location+'dataset/train/spectrogram/GriffinLim/real/data_'+str(label_id)+'_accel_watch.txt'
    
    inv_label = np.empty((3600, 3), dtype=object)
    inv_label[:,0] = label_id
    inv_label[:,1] = activity
    inv_label[:,2] = 0.0
    
    #inv=min_max_scaler.inverse_transform(inv)
    inv_total = np.concatenate([inv_label, inv], axis = 1)
    """ #save inverted real
    try:
        f=open(f, 'a')
        np.savetxt(f,inv_total,delimiter=',',fmt = '%s', newline=';')
        f.close()
    except:
        np.savetxt(f,inv_total,delimiter=',',fmt = '%s', newline=';')
        f.close();
    """
    
    return


def multivariate_data(dataset, step):
    data = []
    start = step
    for i in range(start, len(dataset)+1, step):
        indices = range(i-step, i)
        data.append(dataset[indices])
    return np.array(data)


def create_time_steps(length):
    return list(np.arange(0, length/frequency,1/frequency))

def show_plot(plot_data, title):
    time_steps = create_time_steps(plot_data.shape[0])
    plt.figure()
    plt.title(title)
    if (plot_data.ndim>1):
        plt.plot(time_steps, plot_data[:, 0].flatten(), color='red', label='Accel-X')
        plt.plot(time_steps, plot_data[:, 1].flatten(), color='green', label='Accel-Y')
        plt.plot(time_steps, plot_data[:, 2].flatten(), color='blue', label='Accel-Z')
    else:
        plt.plot(time_steps, plot_data.flatten(), color='red', label='Accel-X')
    plt.legend()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration ($m/s^2$)')
    plt.show()
    #plt.savefig(project_location+'plots/GAN/'+title+'.png')
    return plt

def gen_fake(sumx):
    """ generate x fake data, training for each activity type """
        
    prog = tf.keras.utils.Progbar(50)
    for j in range(1640, 1600+50):
        j=1602

        p_sumx = sumx.loc[(sumx['labels']==str(j))]# | (sumx['labels']=='\n'+str(j))]     #sumx is selected activity data

        for i in range(len(activity)):
            ac = activity[i]
            
            try:
                os.makedirs(project_location+'/dataset/train/spectrogram/test_gan/Activity'+str(activity[i]))
            except FileExistsError:
                # directory already exists
                #print("Dir already exists")
                pass
            
            pre_proc(p_sumx, ac, j)
        return
        
        prog.update(j-1600)
    return

#define activity list - no n data
#activity = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S']
activity = ['A', 'B', 'D', 'Q']
""" [walking, jogging, sitting, writing] """

#min_max_scaler = [range(len(activity))]

#import training data
sumx = import_data()

#for p in range(1651, 1702):
    #os.remove(project_location+'dataset/accel_fake/data_'+str(p)+'_accel_watch.txt')


#train_gan(sumx)
gen_fake(sumx)