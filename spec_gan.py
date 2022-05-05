# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 12:50:00 2022

@author: calum

this file contains code from [1]
and from https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
"""


import tensorflow as tf
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
        
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten #,LSTM
import skimage.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa
from sklearn import preprocessing
import sys

#lowess
import statsmodels.api as sm
#Savitky-Golay
from scipy.signal import savgol_filter

EPOCH=500
N_CONV=3
BATCH_SIZE=10
step=200
frequency=20
# size of the latent space
latent_dim = 100

activity = ['A', 'B', 'D', 'Q']
""" [walking, jogging, sitting, writing] """


data_in = "C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/dataset/train/spectrogram/test_gan/"
offline = "C:/Users/calum/Desktop/"
path = "C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/spectrogram/"
project_location = "C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/"

names=['labels','activities','time series','x', 'y', 'z']

""" data import """
# read specs
# convert to vector: int to float?, reshape, 255-> 1,
# latent dim vector
def import_raw():
    sumx = pd.read_csv(
        r"C:\Users\calum\OneDrive\Documents\Year_3_Project\Python\oldGAN\dataset\train\data_1600_accel_watch.txt"
        #r"C:\Users\calum\OneDrive\Documents\Year_3_Project\Python\oldGAN\dataset\train\data_1600_accel_watch.txt"
        ,names=names, lineterminator=';', skip_blank_lines=True)

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
    
    for i in range(1661,1661):
        filename='C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/oldGAN/dataset/gen_10/data_'+str(i)+'_accel_watch.txt'
        #x=pd.read_csv(filename,names=names)
        #data_raw=pd.concat([data_raw, x], axis=0)
        
    #print(sumx.shape[0])
    
    return sumx

def import_data(activity):
    # channels: 1=greyscale, 3=rgb
    
    act_in = data_in+'Activity'+activity+'/'
    sumx = skimage.io.imread(act_in+str(1600)+'.png')

    sumx = np.expand_dims(sumx, axis = 0) 

    #loop through full set
    for i in range(1601, 1600+50):
        try:
            x = skimage.io.imread(act_in+str(i)+'.png')
        except:
            continue
        x = np.expand_dims(x, axis = 0) 
        sumx = np.concatenate((sumx, x), axis=0)
     #A = 0, B = 1, D = 2, Q = 3
    return sumx

def create_time_steps(length):
    return list(np.arange(0, length/frequency,1/frequency))

def multivariate_data(dataset, step):
    data = []
    start = step
    for i in range(start, len(dataset)+1, step):
        indices = range(i-step, i)
        data.append(dataset[indices])
    return np.array(data)

def show_plot(plot_data, title):
    time_steps = create_time_steps(plot_data.shape[0])
    plt.figure()
    plt.title(title)
    if (plot_data.ndim>1):
        plt.plot(time_steps, plot_data[:, 0].flatten(), color='red', label='Accel-X')
        plt.plot(time_steps, plot_data[:, 1].flatten(), color='blue', label='Accel-Y')
        plt.plot(time_steps, plot_data[:, 2].flatten(), color='yellow', label='Accel-Z')
    else:
        plt.plot(time_steps, plot_data.flatten(), color='red', label='Accel-X')
    plt.legend()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration ($m/s^2$)')
    plt.show()
    #plt.savefig(project_location+'plots/GAN/'+title+'.png')
    return plt

def post_proc(real_x, fake_img, activity, label_id):
    """ retrieve img min max scaler"""
    #only use 3-axis data
    features_considered = ['x', 'y', 'z']
    features=real_x[features_considered]

    dataset = features.values
    
    if (dataset.shape[0] == 0):
        return 0, 0 
    
    if (dataset.shape[0] < 3600):
        print(label_id)
        print("too small")
    
    while (dataset.shape[0] > 3600):
        #print(dataset.shape)
        dataset = dataset[:-1]
        
    while (dataset.shape[0] < 3600):
        
        dataset = np.vstack((dataset, dataset[-1,:] ))
    
    
    x_train_single = multivariate_data(dataset, step)
    
    nfft=256
    hop=int(nfft/4)#28
    hop=28
    
    #fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True)
    Dx = librosa.stft(dataset[:,0], hop_length=hop, n_fft=nfft)  # STFT of x
    Dy = librosa.stft(dataset[:,1], hop_length=hop, n_fft=nfft)  # STFT of x
    Dz = librosa.stft(dataset[:,2], hop_length=hop, n_fft=nfft)  # STFT of x
    
    S_mag = np.empty((129, 129, 3))
    S_mag[:,:,0] = np.abs(Dx)
    S_mag[:,:,1] = np.abs(Dy)
    S_mag[:,:,2] = np.abs(Dz)
    
    S_db = np.empty((129, 129, 3))
    S_db[:,:,0] = librosa.amplitude_to_db(np.abs(Dx), ref=1700)
    S_db[:,:,1] = librosa.amplitude_to_db(np.abs(Dy), ref=1700)
    S_db[:,:,2] = librosa.amplitude_to_db(np.abs(Dz), ref=1700)
    
    S_db=S_db.reshape((129*129, 3))
    img_scaler=preprocessing.MinMaxScaler(feature_range=(0, 255), copy=True).fit(S_db)
    
    
    """ invert """ 
    #[0, 1] -> [0, 255]
    img_inv = fake_img*255.0
    
    img_inv = 255-img_inv
    img_inv = np.flip(img_inv, axis=0)
    img_inv=img_inv.reshape(129*129, 3)
    S_db = img_scaler.inverse_transform(img_inv).reshape((129, 129, 3))
    
    #get db from amplitude
    Dx = librosa.db_to_amplitude(S_db[:,:,0], ref=1700)
    Dy = librosa.db_to_amplitude(S_db[:,:,1], ref=1700)
    Dz = librosa.db_to_amplitude(S_db[:,:,2], ref=1700)
    
    #griffin lim
    invx = librosa.griffinlim(Dx, hop_length=hop, n_fft=nfft, n_iter=16, momentum=1, length=3600, init=None)
    invy = librosa.griffinlim(Dy, hop_length=hop, n_fft=nfft, n_iter=16, momentum=1, length=3600, init=None)
    invz = librosa.griffinlim(Dz, hop_length=hop, n_fft=nfft, n_iter=16, momentum=1, length=3600, init=None)
    
    """ save inverted fake data """
    inv = np.empty((3600, 3))
    inv[:,0] = invx
    inv[:,1] = invy
    inv[:,2] = invz

    #real raw data seems to be clipped at [-20, 20]
    inv = np.clip(inv, -20, 20)
    
    x_train_inv = multivariate_data(inv, step)
    
    if(label_id==1640):
        #show_plot(x_train_single[0,:,0], str(activity)+'_Raw_Min-max_')
        show_plot(x_train_single[0], str(activity)+'_Raw_Min-max_')
        show_plot(x_train_inv[0], str(activity)+'_inv_Min-max_')
        show_plot(dataset, str(activity)+'_Raw_Full_')
        show_plot(inv, str(activity)+'_Inv_Full_')
        #show_plot(inv_f, str(activity)+'_Inv_Full_Filter')
    
    
    #f = project_location+'dataset/train/spectrogram/GriffinLim/real/no_label/Activity'+str(activity)+'/data_'+str(label_id)+'_accel_watch.txt'

    inv_label = np.empty((3600, 3), dtype=object)
    inv_label[:,0] = label_id
    inv_label[:,1] = activity
    inv_label[:,2] = 0.0
    
    #inv=min_max_scaler.inverse_transform(inv)
    
    inv_total = np.concatenate([inv_label, inv], axis = 1)
    
    f = project_location+'dataset/train/spectrogram/GriffinLim/fake/data_'+str(label_id)+'_accel_watch.txt'
    
    try:
        f=open(f, 'a')
        np.savetxt(f,inv_total,delimiter=',',fmt = '%s', newline=';')
        f.close()
    except:
        np.savetxt(f,inv_total,delimiter=',',fmt = '%s', newline=';')
        f.close();
    
    
    return

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
    X = dataset[ix]
	# generate 'real' class labels (1)
    y = np.ones((n_samples, 1))
    return X, y
    
def load_real(activity):
    # add  to image data
    #x_real = np.expand_dims(import_data(activity), axis = -1)  
    x_real = import_data(activity)
    #print(x_real.shape)
    # convert uint8 to float
    x_real = x_real.astype('float32')
    # scale pixel value [0,255] to [0,1]
    #x_real = x_real/255.0
    x_real = (x_real/128.0) - 1
    
    return x_real

# generate latent points
def load_noise(latent_dim, n_samples):
    # generate points in the latent space
	x_noise = np.random.randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_noise = x_noise.reshape(n_samples, latent_dim)
	return x_noise

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = load_noise(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = np.zeros((n_samples, 1))
	return X, y

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def save_fake(X, activity):
    axis = ['x','y','z']
    d_full = import_raw()
    # min-max scale to fit inside 8-bit range
    X = (X + 1) / 2
    for i in range(10):
        out = offline+'dataset/Activity'+str(activity)+'/'+str(1640+i)+'.png'
        
        #does skimage auto convert from [0,1] to [0,255]
        skimage.io.imsave(out, X[i,:,:,:])
        print("/////////////////////////////////////////////////////////")
        post_proc(d_full, X[i,:,:,:], activity, i+1640)
    return

def save_one(X, activity, e):
    axis = ['x','y','z']
    d_full = import_raw()

    out = path+'dataset/spectrogram/Activity'+str(activity)+'/e'+str(e)+'.png'
    #print(X.shape)
        #does skimage auto convert from [0,1] to [0,255]
    skimage.io.imsave(out, X[0,:,:,:])
    return


""" define G """
def generator_model(in_shape=(None, 100)):
    """ full size: 129x129
        half size: 65x65
        quarter size: 33x33
    """
    dim_total = [33, 17, 9, 5]
    #5=5, 4=9, 3=17, 2=33 - need to double check this actually
    dim = dim_total[N_CONV-2]
    kernel_size = 5
    init = keras.initializers.RandomNormal(stddev=0.02)
    
	# foundation for  image
    n_nodes = 2**(5+N_CONV) * dim * dim
    
    model = keras.Sequential()
    model._name="Generator"
    
    #nodes? should it not be 2=1, 3=2, 4=4, 5=8
    
    #(None, 100) -> (None, 139392)
    model.add(layers.Dense(n_nodes, kernel_initializer=init, input_shape=in_shape, name="Fully-Connected"))
    
    for layer in model.layers:
        print(layer.output_shape)
     
    model.add(layers.ReLU())
    model.add(Reshape((dim, dim, 2**(5+N_CONV))))

    if(N_CONV>=5):
        # upsample to 9x9 using transpose
        model.add(layers.Conv2DTranspose(64*8, kernel_size, strides=(2,2), kernel_initializer=init, padding='same', name='TransConv_5', use_bias=False))
        model.add(layers.ReLU())
        model.add(layers.Cropping2D(cropping=((1, 0), (1, 0)), data_format=None, name='crop_to_9'))
    if(N_CONV>=4):    
        # upsample to 17x17 using transpose
        model.add(layers.Conv2DTranspose(64*4, kernel_size, strides=(2,2), kernel_initializer=init, padding='same', name='TransConv_4', use_bias=False))
        model.add(layers.ReLU())
        model.add(layers.Cropping2D(cropping=((1, 0), (1, 0)), data_format=None, name='crop_to_17'))
    if(N_CONV>=3):   
        # upsample to 33x33 using transpose
        model.add(layers.Conv2DTranspose(64*2, kernel_size, strides=(2,2), kernel_initializer=init, padding='same', name='TransConv_3', use_bias=False))
        model.add(layers.ReLU())
        model.add(layers.Cropping2D(cropping=((1, 0), (1, 0)), data_format=None, name='crop_to_33'))
    if(N_CONV>=2):   
        #sample to 65x65 using transpose
        model.add(layers.Conv2DTranspose(64, kernel_size, strides=(2,2), kernel_initializer=init, padding='same', name='TransConv_2', use_bias=False))
        model.add(layers.ReLU())
        model.add(layers.Cropping2D(cropping=((1, 0), (1, 0)), data_format=None, name='crop_to_65'))
	# upsample to 129x129 using transpose    
    model.add(layers.Conv2DTranspose(3, kernel_size, strides=(2,2), kernel_initializer=init, padding='same', name='TransConv_1', use_bias=False))
    model.add(layers.Cropping2D(cropping=((1, 0), (1, 0)), data_format=None, name='crop_to_129'))  
    model.add(layers.Activation('tanh', name='Tanh'))
    
    print(model.summary())
    return model


""" define D 

    filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).

    kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. 
    Can be a single integer to specify the same value for all spatial dimensions.

"""
def discriminator_model(in_shape=(129,129,3)):
    model = keras.Sequential()
    model._name="Discriminator"
    kernel_size = 5
    
    #model = keras.Sequential()
    # input shape (None, 129, 129, 1) -     129 or 128
    init = keras.initializers.RandomNormal(stddev=0.02)

    #(129, 129, 3) -> (65, 65, 65)
    if(N_CONV>=1):
        model.add(layers.Conv2D(64, kernel_size, strides=(2, 2), padding='same', kernel_initializer=init, input_shape=in_shape, name='Conv_1'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.5))
    # (65, 65, 65) -> (33, 33, 65)
    if(N_CONV>=2):
        model.add(layers.Conv2D(64*2, kernel_size, strides=(2, 2),  kernel_initializer=init, padding='same', name='Conv_2'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.5))
    #17
    if(N_CONV>=3):
        model.add(layers.Conv2D(64*4, kernel_size, strides=(2, 2),  kernel_initializer=init, padding='same', name='Conv_3'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.5))
    #9
    if(N_CONV>=4):
        model.add(layers.Conv2D(64*8, kernel_size, strides=(2, 2),  kernel_initializer=init, padding='same', name='Conv_4'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.5))
    #5
    if(N_CONV>=5):
        model.add(layers.Conv2D(64*16, kernel_size, strides=(2, 2),  kernel_initializer=init, padding='same', name='Conv_5'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.5))
        
    model.add(Flatten())
    
    #sigmoid layer - output nodes
    model.add(Dense(1, name='Fully-Connected'))
    model.add(layers.Activation('sigmoid', name='Sigmoid'))


    # compile model
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
    d_model.trainable = False
	# connect them
    model = keras.Sequential()
	# add generator
    model.add(g_model)
	# add the discriminator
    model.add(d_model)
	# compile model
    
    #adam default = 0.001
	#opt = keras.optimizers.Adam(lr=1e-3, beta_1=0.5)
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    #print(model.summary())
    return model

""" train GAN """

# train epochs
# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=50):
	# prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    
    #save_fake(x_fake, epoch)
	# save plot
	#save_plot(x_fake, epoch)
	# save the generator model tile file
	#filename = 'generator_model_%03d.h5' % (epoch + 1)
	#g_model.save(filename)

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
	# plot loss
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(d1_hist, label='d-real')
    plt.plot(d2_hist, label='d-fake')
    plt.plot(g_hist, label='gen')
    plt.legend()
	# plot discriminator accuracy
    plt.subplot(2, 1, 2)
    plt.plot(a1_hist, label='acc-real')
    plt.plot(a2_hist, label='acc-fake')
    plt.legend()
	# save plot to file
	#plt.savefig('results_baseline/plot_line_plot_loss.png')
	#plt.close() 
    
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, act, latent_dim, n_epochs=EPOCH, n_batch=10):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    #print("Dataset.shape")
    #print(dataset.shape)
    half_batch = int(n_batch / 2)
    gen_loss_set=np.empty((n_epochs,1))
    dis_loss_set=np.empty((n_epochs,1))
    epoch_set=np.empty((n_epochs,1))
    
    # prepare lists for storing stats each iteration
    d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
	
    # manually enumerate epochs
    for i in range(n_epochs):
		# enumerate batches over the training set
        for j in range(bat_per_epo):
            
            """ get batch of (real, fake) """
			# get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)
			# generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)
			# create training set for the discriminator
            #print("Testpoint")
            #print(X_real.shape, X_fake.shape)
            #X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
            """ train on batch - single gradient update on a single batch of data.
                recommended to train D 2x per G training (1 for real, 1 for fake?)
                does this fit in that idea? - 1 batch: 2 mini-batch
            """
			# update discriminator model weights
            #d_loss, _ = d_model.train_on_batch(X, y)
			# prepare points in latent space as input for the generator
            X_gan = load_noise(latent_dim, n_batch)
			# create inverted labels for the fake samples (1 = real, 0 = fake)
            y_gan = np.ones((n_batch, 1))
			# update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
            print('>%d, d1=%.3f, d2=%.3f g=%.3f, a1=%d, a2=%d' %
                  (i+1, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))
            gen_loss_set[i]=g_loss
            dis_loss_set[i]=d_loss1
            epoch_set[i]=i+1
        # record history
            #d1_hist.append(d_loss1)
            #d2_hist.append(d_loss2)
            #g_hist.append(g_loss)
            #a1_hist.append(d_acc1)
            #a2_hist.append(d_acc2)
        
		# evaluate the model performance, sometimes
        if (i+1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)
            #x_fake, y_fake = generate_fake_samples(g_model, latent_dim, 1)
            #save_one(x_fake, act, i)
            #g_model.save('C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/spectrogram/models/nconv'+str(N_CONV)+'/spec_g_e'+str(i)+'_'+str(act)) 
        if (i+1) % 50 == 0:
            #g_model.save('C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/spectrogram/models/nconv'+str(N_CONV)+'/spec_g_e'+str(i)+'_'+str(act)) 
            x_fake, y_fake = generate_fake_samples(g_model, latent_dim, 1)
            save_one(x_fake, act, i)
            
            # generate images
            #latent_points = load_noise(100, 5)
            # generate images
            #X = g_model.predict(latent_points)
            #save_fake(X, i)
            
    
    #plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)
    #return
    print(np.mean(dis_loss_set))
    plt.figure()
    plt.plot(epoch_set, gen_loss_set.flatten(), color='red', label='Generator loss')
    plt.plot(epoch_set, dis_loss_set.flatten(), color='blue', label='Discriminator loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Activity '+str(act)+' Loss')
        #plt.savefig(project_location+'plots/GAN/'+str(activity)+'_loss.png')
        #show_plot(pre_time_series[1].numpy(), (str(activity)+'_image_epoch_{:04d}').format (epoch+1))
        #show_plot(pre_time_series[1].numpy(), (str(activity)+'_image_epoch_{:04d}').format (epoch+1))

def run_gan(latent_dim):
    keras.backend.clear_session()
    
    for i in range(len(activity)):
        keras.backend.clear_session()
        e = 499
        #g_model = keras.models.load_model('C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/spectrogram/models/nconv'+str(N_CONV)+'/spec_g_e'+str(e)+'_'+str(activity[i])) 
        
        #i = 3
        # create the discriminator
        d_model = discriminator_model()
        # create the generator
        g_model = generator_model()
        # create the gan
        gan_model = define_gan(g_model, d_model)
        #return
        #i = 3
        dataset = load_real(activity[i])
        
        #x_fake = np.zeros((50,129,129,3))

        train(g_model, d_model, gan_model, dataset, activity[i], latent_dim)
            
        #for j in range(50):
        x_fake_ax, y_fake = generate_fake_samples(g_model, latent_dim, 10)
        save_fake(x_fake_ax, activity[i])
        
        #print(g_model.summary())
        #print(d_model.summary())
        #sys.exit()
        #plot_model(g_model, to_file='C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/spectrogram/models/3_layer_g_model.png', show_shapes=True, show_layer_names=True)
        #plot_model(d_model, to_file='C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/spectrogram/models/3_layer_d_model.png', show_shapes=True, show_layer_names=True)
        #plot_model(gan_model, to_file='C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/spectrogram/models/3_layer_gan_model.png', show_shapes=True, show_layer_names=True, expand_nested=True)
        
        #print_layers(g_model, "z")
        #print_layers(d_model, "G(z) or x")
        #print_layers(gan_model, "z")
        
        g_model.save('C:/Users/calum/OneDrive/Documents/Year_3_Project/Python/spectrogram/models/final/2spec_g_e500_'+str(activity[i])) 
               
    return

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


run_gan(latent_dim)




