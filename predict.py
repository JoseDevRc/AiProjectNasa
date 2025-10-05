import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import normalize
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar modelo ya entrenado
model = load_model("./DataSet/exoplanet_model_2.keras")

# Cargar dataset de prueba
data_test = pd.read_csv("./DataSet/exoTest.csv").to_numpy()

# separar etiqueta y datos
y_test = data_test[:,0]-1   # convertir (1,2) → (0,1)
X_test = data_test[:,1:]

# normalizar como en entrenamiento
X_test_norm = normalize(X_test)

# seleccionar una estrella (ej: índice 42)
idx = 40

star_flux = X_test_norm[idx]

# preparar para el modelo (reshape)
star_flux_nn = star_flux.reshape(1, -1, 1)

# predicción
pred = model.predict(star_flux_nn)
print(f" valor = {pred}")
pred_label = int(pred > 0.5)
print(f" valor2 = {pred_label}")
print(f" valor3 = {y_test[idx]}")


exoplanet_flux = data_test[120, 1:]   # quitamos la etiqueta
label = data_test[120, 0]             # etiqueta real

print("Etiqueta real:", "Exoplanet" if label == 1 else "No Exoplanet")
print("Flux de ejemplo:", exoplanet_flux[:20])  # primeras 20 mediciones

print(f"⭐ Star ID {idx}")
print("Predicción del modelo:", "YES exoplanet" if pred_label == 1 else "NO exoplanet")
print("Etiqueta real:", "YES exoplanet" if y_test[idx] == 1 else "NO exoplanet")
