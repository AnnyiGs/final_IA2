"""
preprocessing.py
─────────────────────────────────────────────────────────────────────
Carga la base de datos MIT-BIH (versión CSV de Kaggle) y aplica:
  1. Carga del CSV
  2. Mapeo de etiquetas
  3. Segmentación / padding a longitud fija
  4. Normalización Min-Max
  5. One-hot encoding de etiquetas
  6. Train/Test split estratificado
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

# ─── Mapeo de clases MIT-BIH ─────────────────────────────────────
# El CSV de Kaggle usa las columnas [0..186] + columna 187 = clase
LABEL_MAP = {
    0: "Normal (N)",
    1: "Supraventricular (S)",
    2: "Ventricular (V)",
    3: "Fusion (F)",
    4: "Desconocido (Q)"
}

SIGNAL_LEN = 180   # longitud fija de cada segmento (descartar col 187)


def cargar_y_preprocesar(
    train_path: str = "mitbih_train.csv",
    test_path:  str = "mitbih_test.csv",
    test_size:  float = 0.20,
    random_state: int = 42
):
    """
    Parámetros
    ----------
    train_path   : ruta al archivo mitbih_train.csv
    test_path    : ruta al archivo mitbih_test.csv
    test_size    : fracción de datos para test (si solo hay train)
    random_state : semilla de aleatoriedad

    Retorna
    -------
    X_train, X_test, y_train, y_test, num_classes, class_names
    """

    # ── Intenta cargar los dos CSV oficiales de Kaggle ────────────
    try:
        df_train = pd.read_csv(train_path, header=None)
        df_test  = pd.read_csv(test_path,  header=None)
        print(f"✔  Archivos cargados: {train_path} ({len(df_train)} muestras) "
              f"y {test_path} ({len(df_test)} muestras)")

        X_train = df_train.iloc[:, :SIGNAL_LEN].values.astype(np.float32)
        y_train_raw = df_train.iloc[:, -1].values.astype(int)

        X_test  = df_test.iloc[:, :SIGNAL_LEN].values.astype(np.float32)
        y_test_raw = df_test.iloc[:, -1].values.astype(int)

    except FileNotFoundError:
        # ── Fallback: datos sintéticos para prueba del código ─────
        print("⚠  Archivos CSV no encontrados. Generando datos SINTÉTICOS para prueba.")
        print("   Descarga el dataset desde: https://www.kaggle.com/datasets/shayanfazeli/heartbeat")
        print("   y coloca mitbih_train.csv / mitbih_test.csv junto a main.py\n")

        np.random.seed(random_state)
        N, num_cls = 1000, 5
        X_all = np.random.randn(N, SIGNAL_LEN).astype(np.float32)
        y_all = np.random.randint(0, num_cls, size=N)

        X_train, X_test, y_train_raw, y_test_raw = train_test_split(
            X_all, y_all, test_size=test_size,
            stratify=y_all, random_state=random_state
        )

    # ── Normalización Min-Max señal por señal ─────────────────────
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Clases presentes ──────────────────────────────────────────
    clases_presentes = sorted(np.unique(y_train_raw))
    num_classes  = len(clases_presentes)
    class_names  = [LABEL_MAP.get(c, str(c)) for c in clases_presentes]

    # Re-indexar etiquetas a 0..num_classes-1
    label_remap = {v: i for i, v in enumerate(clases_presentes)}
    y_train_idx = np.array([label_remap[l] for l in y_train_raw])
    y_test_idx  = np.array([label_remap[l] for l in y_test_raw])

    # ── One-hot encoding ──────────────────────────────────────────
    y_train = to_categorical(y_train_idx, num_classes=num_classes)
    y_test  = to_categorical(y_test_idx,  num_classes=num_classes)

    return X_train, X_test, y_train, y_test, num_classes, class_names
