import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
from scipy.fftpack import fft
import scipy
import seaborn as sns
import models as m

import sklearn.linear_model as lm
import tensorflow as tf
import sklearn.svm as svm

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence

import sklearn.preprocessing as pproc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import normalize
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler



# import the dataset
data_train = pd.read_csv('./DataSet/exoTrain.csv')
#data_test = pd.read_csv('./DataSet/exoTest.csv')



#permute the dataset
data_train = np.random.permutation(np.asarray(data_train))
#data_test = np.random.permutation(np.asarray(data_test))



#get the Label column and delate the class column and rescale
y1 = data_train[:,0]
#y2 = data_test[:,0]

y_train = (y1-min(y1))/(max(y1)-min(y1))
#y_test = (y2-min(y2))/(max(y2)-min(y2))

data_train = np.delete(data_train,1,1)
#data_test = np.delete(data_test,1,1)



#print the light curve
time = np.arange(len(data_train[0])) * (36/60)  # time in hours

plt.figure(figsize=(20,5))
plt.title('Flux of star 10 with confirmed planet')
plt.ylabel('Flux')
plt.xlabel('Hours')
plt.plot( time , data_train[10] )     #change the number to plot what you want



#normalized data
data_train_norm = normalize(data_train)
#data_test_norm = normalize(data_test)




# function to apply gaussian filter to all data
def gauss_filter(dataset,sigma):
    
    dts = []
    
    for x in range(dataset.shape[0]):
        dts.append(gaussian_filter(dataset[x], sigma))
    
    return np.asarray(dts)



# apply the gaussian filter to all rows data
data_train_gaussian = gauss_filter(data_train_norm,7.0)
#data_test_gaussian = gauss_filter(data_test_norm,7.0)




#print the light curves smoothed
plt.figure(figsize=(20,5))
plt.title('Flux of star 10 with confirmed planet, smoothed')
plt.ylabel('Flux')
plt.xlabel('Hours')
#plt.plot( time , data_train_gaussian[1000])




# apply FFT to the data smoothed
frequency = np.arange(len(data_train[0])) * (1/(36.0*60.0))

data_train_fft1 = scipy.fft.fft2(data_train_norm, axes=1)
#data_test_fft1 = scipy.fft.fft2(data_test_norm, axes=1)

data_train_fft = np.abs(data_train_fft1)   #calculate the abs value
#data_test_fft = np.abs(data_test_fft1)



#get the length of the FFT data, make something here below in order to make the sequences of the same size
# only if they have differet dimensions

len_seq = len(data_train_fft[0])



#plot the FFT of the signals
plt.figure(figsize=(20,5))
plt.title('flux of star 1 ( with confirmed planet ) in domain of frequencies')
plt.ylabel('Abs value of FFT result')
plt.xlabel('Frequency')
plt.plot( frequency, data_train_fft[1] )



#oversampling technique to the data
smote = SMOTE(random_state=42)
data_train_ovs, y_train_ovs = smote.fit_resample(data_train_fft, y_train)



#recap dataset after oversampling
print("After oversampling, counts of label '1': {}".format(sum(y_train_ovs==1)))
print("After oversampling, counts of label '0': {}".format(sum(y_train_ovs==0)))



#reshape the data for the neural network model
data_train_ovs = np.asarray(data_train_ovs)
#data_test_fft = np.asarray(data_test_fft)

data_train_ovs_nn = data_train_ovs.reshape((data_train_ovs.shape[0], data_train_ovs.shape[1], 1))
#data_test_fft_nn = data_test_fft.reshape((data_test_fft.shape[0], data_test_fft.shape[1], 1))



#create F.C.N model and run it
model = m.FCN_model(len_seq)

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])

print(model.summary())

history = model.fit(data_train_ovs_nn, y_train_ovs , epochs=15, batch_size = 10, validation_split=0.2)


#save the model if you want
model.save("./DataSet/exoplanet_model_2.keras")