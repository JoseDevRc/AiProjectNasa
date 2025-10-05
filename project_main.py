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
from sklearn.utils import class_weight


data_train = pd.read_csv('./DataSet/exoTrain.csv')
data_test = pd.read_csv('./DataSet/exoTest.csv')



data_train = np.random.permutation(np.asarray(data_train)) #Este es el que entrenara el modelo
data_test = np.random.permutation(np.asarray(data_test))#Este es el que testeara el modelo



y1 = data_train[:,0]
y2 = data_test[:,0]

y_train = (y1-min(y1))/(max(y1)-min(y1))
y_test = (y2-min(y2))/(max(y2)-min(y2))

data_train = np.delete(data_train,1,1)
data_test = np.delete(data_test,1,1)




time = np.arange(len(data_train[0])) * (36/60)  # timepo en horas
numb = 1 #Cambia el número para graficar cualquier estrella de la grafica

plt.figure(figsize=(20,5))
plt.title('Flux of star 1 with confirmed planet')
plt.ylabel('Flux')
plt.xlabel('Hours')
plt.plot( time , data_train[numb] )     



#normalizar datos
data_train_norm = normalize(data_train)
data_test_norm = normalize(data_test)




# Función para aplicar el filtro gaussiano a todos los datos
def gauss_filter(dataset,sigma):
    
    dts = []
    
    for x in range(dataset.shape[0]):
        dts.append(gaussian_filter(dataset[x], sigma))
    
    return np.asarray(dts)



# Aplicar el filtro gaussiano a todas las filas de datos
data_train_gaussian = gauss_filter(data_train_norm,7.0)
data_test_gaussian = gauss_filter(data_test_norm,7.0)




#Imprimir (o mostrar) las curvas de luz suavizadas
plt.figure(figsize=(20,5))
plt.title(f'Flux of star {numb} with confirmed planet, smoothed')
plt.ylabel('Flux')
plt.xlabel('Hours')
plt.plot( time , data_train_gaussian[1000])




# apply FFT to the data smoothed
frequency = np.arange(len(data_train[0])) * (1/(36.0*60.0))

data_train_fft1 = scipy.fft.fft2(data_train_norm, axes=1)
data_test_fft1 = scipy.fft.fft2(data_test_norm, axes=1)

data_train_fft = np.abs(data_train_fft1)   
data_test_fft = np.abs(data_test_fft1)




len_seq = len(data_train_fft[0])



#plot the FFT of the signals
plt.figure(figsize=(20,5))
plt.title(f'flux of star {numb} ( with confirmed planet ) in domain of frequencies')
plt.ylabel('Abs value of FFT result')
plt.xlabel('Frequency')
plt.plot( frequency, data_train_fft[1] )



smote = SMOTE(random_state=42)
data_train_ovs, y_train_ovs = smote.fit_resample(data_train_fft, y_train)


print("After oversampling, counts of label '1': {}".format(sum(y_train_ovs==1)))
print("After oversampling, counts of label '0': {}".format(sum(y_train_ovs==0)))




data_train_ovs = np.asarray(data_train_ovs)
data_test_fft = np.asarray(data_test_fft)

data_train_ovs_nn = data_train_ovs.reshape((data_train_ovs.shape[0], data_train_ovs.shape[1], 1))
data_test_fft_nn = data_test_fft.reshape((data_test_fft.shape[0], data_test_fft.shape[1], 1))


class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_ovs),
    y=y_train_ovs
)
class_weights = dict(enumerate(class_weights))
print("Pesos de clase:", class_weights)

#crea el modelo F.C.N, solo si no tienes el modelo y descomentalo si no lo tienes tambien
#model = m.FCN_model(len_seq)

#model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])

#print(model.summary())

#history = model.fit(data_train_ovs_nn, y_train_ovs , epochs=15, batch_size = 10, validation_data=(data_test_fft_nn, y_test),class_weight=class_weights)


#descomenta este para guardar el modelo
#model.save("./DataSet/exoplanet_model_2.keras")


#carga el modelo si ya existe, si no lo tienes comentalo
model = tf.keras.models.load_model("./DataSet/exoplanet_model_2.keras")

#descomenta si no tienes el modelo creado

#acc = history.history['accuracy']
#acc_val = history.history['val_accuracy']
#epochs = range(1, len(acc)+1)
#plt.plot(epochs, acc, 'b', label='accuracy_train')
#plt.plot(epochs, acc_val, 'g', label='accuracy_val')
plt.title('accuracy')
plt.xlabel('epochs')
plt.ylabel('value of accuracy')
plt.legend()
plt.grid()
plt.show()


#descomenta si no tienes el modelo creado

#loss = history.history['loss']
#loss_val = history.history['val_loss']
#epochs = range(1, len(acc)+1)
#plt.plot(epochs, loss, 'b', label='loss_train')
#plt.plot(epochs, loss_val, 'g', label='loss_val')
#plt.title('loss')
#plt.xlabel('epochs')
#plt.ylabel('value of loss')
#plt.legend()
#plt.grid()
#plt.show()




#Predecir
y_test_pred = model.predict(data_test_fft_nn)
y_test_pred = (y_test_pred > 0.3).astype(int).flatten()


accuracy = accuracy_score(y_test, y_test_pred)
print("accuracy : ", accuracy)

print(classification_report(y_test, y_test_pred, target_names=["NO exoplanet confirmed","YES exoplanet confirmed"]))

conf_matrix = confusion_matrix([int(x) for x in y_test ], [int(y) for y in y_test_pred ])
sns.heatmap(conf_matrix, annot=True, cmap='Blues')




SVC = m.SVC_model()
SVC.fit(data_train_ovs, y_train_ovs)

y_pred_svc = SVC.predict(data_test_fft)


print(classification_report(y_test, y_pred_svc, target_names=["NO exoplanet confirmed","YES exoplanet confirmed"]))


conf_matrix = confusion_matrix([int(x) for x in y_test ], [int(y) for y in y_pred_svc ])
sns.heatmap(conf_matrix, annot=True, cmap='Blues')


resultados = pd.DataFrame({
    "Star_ID": np.arange(len(y_test_pred)),  
    "Prediction_NN": y_test_pred,             
    "Prediction_SVC": y_pred_svc,
    "Real_Label": y_test.astype(int) 
})


print(resultados.head(571))

# Guardar en CSV
resultados.to_csv("Resultados_Predicciones.csv", index=False)

print("\n✅ Resultados guardados en 'Resultados_Predicciones.csv'")

