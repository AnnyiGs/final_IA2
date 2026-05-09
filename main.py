"""
=====================================================================
Proyecto: Clasificación de Arritmias - MIT-BIH
Universidad de Guadalajara - CUCEI
Seminario de Solución de Problemas de IA II
Dr. Diego Oliva

Modelos: ANN | CNN | RNN (LSTM)
=====================================================================
"""

import warnings
warnings.filterwarnings("ignore")

from preprocessing import cargar_y_preprocesar
from models import crear_ann, crear_cnn, crear_rnn
from train import entrenar_modelo
from evaluation import evaluar_modelo, comparar_modelos

# ─────────────────────────────────────────
# 1. PREPROCESAMIENTO
# ─────────────────────────────────────────
print("=" * 60)
print("  CARGANDO Y PREPROCESANDO DATOS MIT-BIH")
print("=" * 60)

X_train, X_test, y_train, y_test, num_classes, class_names = cargar_y_preprocesar()

print(f"\n✔  Clases detectadas : {num_classes} → {class_names}")
print(f"✔  Train shape       : {X_train.shape}")
print(f"✔  Test  shape       : {X_test.shape}\n")

# ─────────────────────────────────────────
# 2. DEFINICIÓN Y ENTRENAMIENTO DE MODELOS
# ─────────────────────────────────────────
input_dim = X_train.shape[1]   # longitud de cada segmento

resultados = {}

# ── ANN ──────────────────────────────────
print("=" * 60)
print("  PRUEBA 1: ANN (Red Neuronal Artificial)")
print("=" * 60)
ann = crear_ann(input_dim, num_classes)
ann.summary()
history_ann, ann = entrenar_modelo(ann, X_train, y_train, X_test, y_test,
                                   model_name="ANN", epochs=50, batch_size=64)
resultados["ANN"] = evaluar_modelo(ann, X_test, y_test,
                                   history_ann, class_names, model_name="ANN")

# ── CNN ──────────────────────────────────
print("=" * 60)
print("  PRUEBA 2: CNN (Red Neuronal Convolucional)")
print("=" * 60)
# CNN espera (samples, timesteps, 1)
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

cnn = crear_cnn(input_dim, num_classes)
cnn.summary()
history_cnn, cnn = entrenar_modelo(cnn, X_train_cnn, y_train, X_test_cnn, y_test,
                                   model_name="CNN", epochs=50, batch_size=64)
resultados["CNN"] = evaluar_modelo(cnn, X_test_cnn, y_test,
                                   history_cnn, class_names, model_name="CNN")

# ── RNN (LSTM) ───────────────────────────
print("=" * 60)
print("  PRUEBA 3: RNN-LSTM (Red Neuronal Recurrente)")
print("=" * 60)
WINDOW = 10   # tamaño de ventana temporal (timesteps)
# Reshape: (samples, timesteps, features)
X_train_rnn = X_train.reshape(X_train.shape[0], WINDOW, input_dim // WINDOW)
X_test_rnn  = X_test.reshape(X_test.shape[0],  WINDOW, input_dim // WINDOW)

rnn = crear_rnn(WINDOW, input_dim // WINDOW, num_classes)
rnn.summary()
history_rnn, rnn = entrenar_modelo(rnn, X_train_rnn, y_train, X_test_rnn, y_test,
                                   model_name="RNN", epochs=50, batch_size=64)
resultados["RNN"] = evaluar_modelo(rnn, X_test_rnn, y_test,
                                   history_rnn, class_names, model_name="RNN")

# ─────────────────────────────────────────
# 3. COMPARACIÓN FINAL
# ─────────────────────────────────────────
comparar_modelos(resultados)
