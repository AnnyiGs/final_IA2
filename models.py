"""
models.py
─────────────────────────────────────────────────────────────────────
Define las tres arquitecturas de redes neuronales:
  · crear_ann  → Red Neuronal Artificial (Fully Connected)
  · crear_cnn  → Red Neuronal Convolucional 1D
  · crear_rnn  → Red Neuronal Recurrente con capas LSTM
─────────────────────────────────────────────────────────────────────
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten,
    Conv1D, MaxPooling1D, BatchNormalization,
    LSTM, Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# ─────────────────────────────────────────────────────────────────
# PRUEBA 1: ANN – Red Neuronal Artificial (Fully Connected)
# ─────────────────────────────────────────────────────────────────
def crear_ann(input_dim: int, num_classes: int,
              lr: float = 1e-3) -> Sequential:
    """
    Arquitectura:
      Input  →  Dense(256, ReLU) → Dropout(0.3)
             →  Dense(128, ReLU) → Dropout(0.3)
             →  Dense(64,  ReLU) → Dropout(0.2)
             →  Dense(num_classes, Softmax)
    """
    model = Sequential(name="ANN", layers=[
        Dense(256, activation="relu", input_shape=(input_dim,),
              kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(128, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(64,  activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─────────────────────────────────────────────────────────────────
# PRUEBA 2: CNN – Red Neuronal Convolucional 1D
# ─────────────────────────────────────────────────────────────────
def crear_cnn(input_dim: int, num_classes: int,
              lr: float = 1e-3) -> Sequential:
    """
    Arquitectura:
      Input  →  Conv1D(32)  → BatchNorm → ReLU → MaxPool → Dropout
             →  Conv1D(64)  → BatchNorm → ReLU → MaxPool → Dropout
             →  Conv1D(128) → BatchNorm → ReLU → MaxPool → Dropout
             →  Flatten
             →  Dense(128, ReLU) → Dropout
             →  Dense(num_classes, Softmax)
    """
    model = Sequential(name="CNN", layers=[
        # Bloque 1
        Conv1D(32, kernel_size=5, padding="same",
               activation="relu", input_shape=(input_dim, 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        # Bloque 2
        Conv1D(64, kernel_size=5, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        # Bloque 3
        Conv1D(128, kernel_size=3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        # Clasificador
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─────────────────────────────────────────────────────────────────
# PRUEBA 3: RNN – Red Neuronal Recurrente (LSTM)
# ─────────────────────────────────────────────────────────────────
def crear_rnn(timesteps: int, features: int, num_classes: int,
              lr: float = 1e-3) -> Sequential:
    """
    Arquitectura:
      Input  →  LSTM(128, return_sequences=True) → Dropout(0.3)
             →  LSTM(64,  return_sequences=True) → Dropout(0.3)
             →  LSTM(32)                          → Dropout(0.2)
             →  Dense(64, ReLU)
             →  Dense(num_classes, Softmax)
    """
    model = Sequential(name="RNN_LSTM", layers=[
        LSTM(128, return_sequences=True,
             input_shape=(timesteps, features)),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
