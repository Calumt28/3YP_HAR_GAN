# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:01:14 2022

@author: calum
"""

#### the code for CNN model####  
import tensorflow as tf 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import confusion_matrix 
import os
import sys
import collections


project_location = "C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/"
#project_locationALT = "C:\Users\calum\OneDrive\Documents\Year_3_Project\Python\oldGAN\"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.getcwd()

names=['labels','activities','time series','x', 'y', 'z']
#activity = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S']
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


# input what data to train CNN
#real:fake split %,       possibly: shuffle, data location  

# train CNN
# save CNN

def import_real(x_split): # x_split datasets, starting at x_start
    
    if (x_split == 0):
        return
    #could introduce randomness here?
    x_start = 1600
    data_raw = pd.read_csv(
        r"C:\Users\calum\OneDrive\Documents\Year_3_Project\Python\oldGAN\dataset\train\data_"+str(x_start)+"_accel_watch.txt"
        ,names=names, lineterminator=';', skip_blank_lines=True)

    data_raw = data_raw[:-1]
    data_raw['labels']=str(x_start)
    
    for i in range(x_start+1,x_start+x_split): #or 1651
        #print(i)
        filename=project_location+'/dataset/train/data_'+str(i)+'_accel_watch.txt'
        x=pd.read_csv(filename,names=names, lineterminator=';')
        x['labels']=str(i)
        data_raw=pd.concat([data_raw, x[:-1]], axis=0)
    
    return data_raw

def import_real_test(): #10 datasets
    return import_real(40, 10)

def import_fake(x_split): # x_split datasets, starting at x_start
    
    if (x_split == 0):
        return
    
    #load 10 pre done
    data_raw = pre_fake()
    if (x_split == 10):
        return data_raw
    #could introduce randomness here?
    x_start = 1651
    data_raw = pd.read_csv(
        r"C:\Users\calum\OneDrive\Documents\Year_3_Project\Python\oldGAN\dataset\accel_fake5\data_"+str(x_start)+"_accel_watch.txt"
        ,names=names, skip_blank_lines=True)
    
    for i in range(x_start+1,x_start+x_split-10):
        
        filename=project_location+'dataset/accel_fake5/data_'+str(i)+'_accel_watch.txt'
        x=pd.read_csv(filename,names=names)
        data_raw=pd.concat([data_raw, x], axis=0)
       
    return data_raw

def pre_fake():
    data_raw = pd.read_csv(
        r"C:\Users\calum\OneDrive\Documents\Year_3_Project\Python\oldGAN\dataset\gen_10\data_1660_accel_watch.txt"
        ,names=names, skip_blank_lines=True)
    #np.savetxt('C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/sumx.txt',data_raw,delimiter=',',fmt = '%s')
    
    for i in range(1661,1660+10):
        filename='C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/dataset/gen_10/data_'+str(i)+'_accel_watch.txt'
        x=pd.read_csv(filename,names=names)
        data_raw=pd.concat([data_raw, x], axis=0)
    
    return data_raw

def my_fake():
    data_raw = pd.read_csv(
        r"C:\Users\calum\OneDrive\Documents\Year_3_Project\Python\oldGAN\dataset\accel_fake5\data_1692_accel_watch.txt"
        ,names=names, skip_blank_lines=True)
    #np.savetxt('C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/sumx.txt',data_raw,delimiter=',',fmt = '%s')
    
    for i in range(1693,1692+10):
        filename='C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/dataset/accel_fake5/data_'+str(i)+'_accel_watch.txt'
        x=pd.read_csv(filename,names=names)
        data_raw=pd.concat([data_raw, x], axis=0)
    
    return data_raw


def multivariate_data(dataset, step):
    data = []
    start = step
    for i in range(start, len(dataset)+1, step):
        indices = range(i-step, i)
        data.append(dataset[indices])
    return np.array(data)


def pre_proc(data_raw):
    data_full=data_raw
    #data_raw=data_raw.loc[data_raw['activities']==activity]     #selected activity data

    data_full=data_full.loc[((data_full['activities'] == 'A') | (data_full['activities'] == 'B')
          | (data_full['activities'] == 'D') | (data_full['activities'] == 'Q'))]
 
    data_clean = pd.DataFrame(columns=names)
    
    l_1 = int(data_full['labels'].values[1])
    for i in range(50):
        
        if(i!=16):
            d_sel=data_full.loc[((data_full['labels'] == str(l_1+i)))]
           
        else:
            d_sel=data_full.loc[((data_full['labels'] == str(l_1)))]
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
    scaler=preprocessing.MinMaxScaler(feature_range=(-1,1)) 
    data=scaler.fit_transform(data)  
    data=pd.DataFrame(data, columns=['x','y','z']) 
    ## Concat label and data 
    label=label.reset_index(drop=True) 
    data=data.reset_index(drop=True) 
    dataset=pd.concat([label,data],axis=1) 
    dataset=np.array(dataset)

    # ###if L2 normalization,uncomment the code before END
    # dataset = tf.keras.utils.normalize(dataset, axis=0,order=2)#L2
    # ###END
    
    x_train_single = multivariate_data(dataset, step)
    #print(dataset)
    
    #datasets is a tf dataset with a single element of shape (batch_size, step, 3)
    #datasets = tf.data.Dataset.from_tensor_slices(x_train_single)
    #randomly shuffles elements of this dataset
    #datasets = datasets.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
     
    #seperate data and labels
    data_n=x_train_single[:,:,1:]  
    label_n=x_train_single[:,0,0] 
    
    #one hot encoding
    label_n=tf.keras.utils.to_categorical(label_n) 
    
        
    return data_n, label_n
    

def create_time_steps(length):
    return list(np.arange(0, length/frequency,1/frequency))


from tensorflow.keras import backend as K

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


## Window segment 
def window_seg(s_total, step):
    data_1=[] 
    start=step  
    for i in range(start, len(s_total)+1, step):  
        indice=range(i-step, i) 
        data_1.append(s_total[indice])  
    return np.array(data_1) 

def test_model(model, d_value, d_label, real=False):
    label_test=d_label
    
    data_test, label_test = d_value,d_label
    #print(d_label.shape, d_value.shape)
    #sys.exit()
    
    #test model on test dataset
    predictions=model.predict(data_test) 
    
    #get array of test labels
    max_true = np.argmax(label_test, axis = 1)

    max_prediction = np.argmax(predictions, axis = 1)
    
    #total is number of entries in test dataset
    correct = 0
    total = max_prediction.shape[0]  
    #print(total)
    
    #set all activity accuracy to 0 by default
    #[correct_n, incorrect_n]
    activity_accuracy = np.zeros((ACT_N, 2))
    
    #loop through set of predictions, comparing predicted label to true label
    for i in range(total):
        if max_true[i] == max_prediction[i]:
            correct = correct + 1
            #increment correct for this activity
            activity_accuracy[max_true[i], 0] = activity_accuracy[max_true[i], 0] + 1
        else:
            #increment incorrect
            activity_accuracy[max_true[i], 1] = activity_accuracy[max_true[i], 1] + 1
    
    total_accuracy = (correct/total)
    
    #print(str(correct)+'/'+str(total)+' = '+str(round(100*correct/total, 2))+'%')
    
    #set accuracy based off correct/total
    for i in range(ACT_N):
        activity_accuracy[i, 0] = activity_accuracy[i, 0] / (activity_accuracy[i, 0]+activity_accuracy[i, 1])   
    """    
    CM = confusion_matrix(max_true, max_prediction) 
    plt.figure(figsize=(16,14)) 
    sns.heatmap(CM, xticklabels = activity, yticklabels = activity, 
    annot = True, fmt = 'd',cmap='Blues') 
    plt.title('Confusion Matrix') 
    plt.xlabel('Predicted Label') 
    plt.ylabel('True Label') 
    plt.show()  
    """
    #return table of activity accuracy + total accuracy
    #acc_res = np.append(activity_accuracy[:,0])
    
    return np.min(activity_accuracy[:,0]), np.mean(activity_accuracy[:,0])



def train_CNN(data_train,data_test,label_train,label_test): #data_split input as integer? {100, 75, 50, 25, 0}

    # CNN model 
    time_steps = data_train.shape[1] 
    features = data_train.shape[2] 
    model = tf.keras.Sequential() 
    model.add(tf.keras.layers.Conv1D(filters=32,kernel_size=5,activation='relu',input_shape=(time_steps,features))) 
    #model.add(tf.keras.layers.BatchNormalization()) 
    model.add(tf.keras.layers.Conv1D(32,5,activation='relu')) 
    #model.add(tf.keras.layers.BatchNormalization()) 
    model.add(tf.keras.layers.Conv1D(64,5,activation='relu')) 
    model.add(tf.keras.layers.Dropout(0.2)) 
    model.add(tf.keras.layers.Conv1D(64,5,activation='tanh')) 
    model.add(tf.keras.layers.MaxPool1D(pool_size=2,strides=2)) 
    model.add(tf.keras.layers.BatchNormalization()) 
    model.add(tf.keras.layers.Dropout(0.5))  
    model.add(tf.keras.layers.Flatten()) 
    model.add(tf.keras.layers.Dense(512,activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))) 
    model.add(tf.keras.layers.Dense(4,activation='softmax'))  
        
    # model compiler settings 
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy',f1_m,precision_m, recall_m])
    #metric = tfa.metrics.F1Score(num_classes=3, threshold=0.5)

    # training Model 
    history=model.fit(data_train,label_train,epochs=200, batch_size=64, 
    validation_split=0.3,shuffle=True,verbose=0)

    # evaluate model 
    #test_loss, test_acc, f1_score, precision, recall=model.evaluate(data_test,label_test,batch_size=64,verbose=0)  
    #print('...test_acc=',test_acc) 
    #print('test_loss=',test_loss) 
    #model.summary()
    med, test_acc = test_model(model, data_test, label_test)

    #save model
    #model.save(project_location+'/models/crossfold/CNN_'+str(data_split)) 
    
    return model, test_acc, history, med 

def plot_graphs(history,metric): 
    plt.plot(history.history[metric])  
    plt.plot(history.history['val_'+metric],'')  
    plt.title('Training and Validation '+metric.capitalize()) #uppercase the first letter  plt.xlabel("Epochs")  
    plt.ylabel(metric) 
    plt.legend([metric,'val_'+metric])
    plt.show()  


def cnn_loop():
    """ load datasets """    
    #data_train, label_train = pre_proc(import_real(40))
    data_n, label_n = pre_proc(import_real(51))
    #print(collections.Counter(label_n))
    
    data_train,data_test,label_train,label_test=train_test_split(data_n,label_n,train_size=0.8, shuffle=False)
    
    #check balanced classes
    #print(data_n.shape, label_train.shape)
    #print(np.sum(label_train[:,0]), np.sum(label_train[:,1]), np.sum(label_train[:,2]), np.sum(label_train[:,3]))
    #sys.exit()
    #shuffle training data
    rng_state = np.random.get_state()
    np.random.shuffle(data_train)
    np.random.set_state(rng_state)
    np.random.shuffle(label_train)
    
    #18552 = 3312 + 15240
    #print(data_n.shape)
    #sys.exit()
    """ train model """
    #data_train=np.random.shuffle(data_train)
    ## divide training set and test set (8:2)  - not testing in this file
    #data_train,data_test,label_train,label_test=train_test_split(d_value,d_label,test_size=0, shuffle=True)   #shuffle to false, no test data so size = 0
    #data_train, label_train = d_value, d_label
    e=50
    progbar = tf.keras.utils.Progbar(e)
    max_acc = 0
    for i in range(e):
        progbar.update(i)
        model, cnn_acc, history, min_acc = train_CNN(data_train,data_test,label_train,label_test)
        if min_acc > max_acc:
            print("Acc: %f, Med: %f", cnn_acc, min_acc)
            max_acc=min_acc
            
            model.save(project_location+'/models/cnn_opt/CNN40RR_clean2') 
            plot_graphs(history,'accuracy') 
            plot_graphs(history,'loss')
        
    return

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
cnn_loop()
