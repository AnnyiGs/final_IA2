import warnings
warnings.filterwarnings("ignore")
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, recall_score, precision_score
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization, LSTM
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

LABEL_MAP = {
    0: "Normal (N)",
    1: "Supraventricular (S)",
    2: "Ventricular (V)",
    3: "Fusion (F)",
    4: "Desconocido (Q)"
}
SIGNAL_LEN = 180
COLORS = {"ANN": "#378ADD", "CNN": "#1D9E75", "RNN": "#D85A30"}

def cargar_y_preprocesar(train_path="DATASET-ECG/mitbih_train.csv", test_path="DATASET-ECG/mitbih_test.csv", test_size=0.20, random_state=42):
    try:
        df_train = pd.read_csv(train_path, header=None)
        df_test = pd.read_csv(test_path, header=None)
        print(f"Archivos cargados: {len(df_train)} muestras train, {len(df_test)} muestras test")
        X_train = df_train.iloc[:, :SIGNAL_LEN].values.astype(np.float32)
        y_train_raw = df_train.iloc[:, -1].values.astype(int)
        X_test = df_test.iloc[:, :SIGNAL_LEN].values.astype(np.float32)
        y_test_raw = df_test.iloc[:, -1].values.astype(int)
    except FileNotFoundError:
        print("Archivos CSV no encontrados. Generando datos sintéticos.")
        np.random.seed(random_state)
        N, num_cls = 1000, 5
        X_all = np.random.randn(N, SIGNAL_LEN).astype(np.float32)
        y_all = np.random.randint(0, num_cls, size=N)
        X_train, X_test, y_train_raw, y_test_raw = train_test_split(
            X_all, y_all, test_size=test_size, stratify=y_all, random_state=random_state
        )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clases_presentes = sorted(np.unique(y_train_raw))
    num_classes = len(clases_presentes)
    class_names = [LABEL_MAP.get(c, str(c)) for c in clases_presentes]
    label_remap = {v: i for i, v in enumerate(clases_presentes)}
    y_train_idx = np.array([label_remap[l] for l in y_train_raw])
    y_test_idx = np.array([label_remap[l] for l in y_test_raw])
    y_train = to_categorical(y_train_idx, num_classes=num_classes)
    y_test = to_categorical(y_test_idx, num_classes=num_classes)

    return X_train, X_test, y_train, y_test, num_classes, class_names

def crear_ann(input_dim, num_classes, lr=1e-3):
    model = Sequential(name="ANN", layers=[
        Dense(256, activation="relu", input_shape=(input_dim,), kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(128, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def crear_cnn(input_dim, num_classes, lr=1e-3):
    model = Sequential(name="CNN", layers=[
        Conv1D(32, kernel_size=5, padding="same", activation="relu", input_shape=(input_dim, 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(64, kernel_size=5, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(128, kernel_size=3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def crear_rnn(timesteps, features, num_classes, lr=1e-3):
    model = Sequential(name="RNN_LSTM", layers=[
        LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def entrenar_modelo(model, X_train, y_train, X_val, y_val, model_name="model", epochs=50, batch_size=64):
    os.makedirs("checkpoints", exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=f"checkpoints/{model_name}_best.keras", monitor="val_accuracy", save_best_only=True, verbose=0)
    ]
    print(f"\n  Entrenando {model_name} (max {epochs} épocas, batch={batch_size})…\n")
    t0 = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
    elapsed = time.time() - t0
    history.history["training_time"] = elapsed
    print(f"\n  {model_name} entrenado en {elapsed:.1f}s | Mejor val_accuracy = {max(history.history['val_accuracy']):.4f}\n")
    return history, model

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    os.makedirs("resultados", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names)-1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, linewidths=0.5, ax=ax)
    ax.set_title(f"Matriz de Confusión — {model_name}", fontsize=14, pad=12)
    ax.set_xlabel("Predicción", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"resultados/{model_name}_confusion_matrix.png", dpi=150)
    plt.show()

def plot_history(history, model_name):
    os.makedirs("resultados", exist_ok=True)
    hist = history.history
    color = COLORS.get(model_name, "#333333")
    epochs = range(1, len(hist["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Evolución del Entrenamiento — {model_name}", fontsize=13)
    axes[0].plot(epochs, hist["accuracy"], color=color, label="Train Accuracy")
    axes[0].plot(epochs, hist["val_accuracy"], color=color, label="Val Accuracy", linestyle="--", alpha=0.8)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, hist["loss"], color=color, label="Train Loss")
    axes[1].plot(epochs, hist["val_loss"], color=color, label="Val Loss", linestyle="--", alpha=0.8)
    axes[1].set_title("Pérdida (Loss)")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"resultados/{model_name}_history.png", dpi=150)
    plt.show()

def evaluar_modelo(model, X_test, y_test, history, class_names, model_name="model"):
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    val_loss = min(history.history["val_loss"])
    t_time = history.history.get("training_time", 0)
    print(f"\n{'─'*55}")
    print(f"  Resultados — {model_name}")
    print(f"{'─'*55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  Val Loss  : {val_loss:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=class_names, zero_division=0)}")
    plot_confusion_matrix(y_true, y_pred, class_names, model_name)
    plot_history(history, model_name)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "val_loss": val_loss, "time": t_time}

def comparar_modelos(resultados):
    modelos = list(resultados.keys())
    metricas = ["accuracy", "precision", "recall", "f1", "val_loss", "time"]
    labels = ["Accuracy", "Precision", "Recall", "F1-score", "Val Loss", "Tiempo (s)"]
    print("\n" + "=" * 65)
    print("  COMPARACIÓN FINAL: ANN  vs  CNN  vs  RNN")
    print("=" * 65)
    header = f"  {'Métrica':<15}" + "".join(f"{m:>12}" for m in modelos)
    print(header)
    print("  " + "-" * (13 + 12 * len(modelos)))
    for met, lab in zip(metricas, labels):
        row = f"  {lab:<15}"
        for m in modelos:
            val = resultados[m][met]
            row += f"{val:>12.4f}" if met != "time" else f"{val:>11.1f}s"
        print(row)
    metricas_plot = ["accuracy", "precision", "recall", "f1"]
    labels_plot = ["Accuracy", "Precision", "Recall", "F1-score"]
    x = np.arange(len(metricas_plot))
    ancho = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, modelo in enumerate(modelos):
        vals = [resultados[modelo][m] for m in metricas_plot]
        bars = ax.bar(x + i * ancho, vals, ancho, label=modelo, color=list(COLORS.values())[i % len(COLORS)])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + ancho)
    ax.set_xticklabels(labels_plot)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Puntuación")
    ax.set_title("Comparación de Métricas: ANN vs CNN vs RNN")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("resultados/comparacion_modelos.png", dpi=150)
    plt.show()
    mejor = max(resultados, key=lambda m: resultados[m]["accuracy"])
    print(f"\n  Mejor modelo por Accuracy: {mejor} ({resultados[mejor]['accuracy']:.4f})\n")

if __name__ == "__main__":
    print("=" * 60)
    print("  CARGANDO Y PREPROCESANDO DATOS MIT-BIH")
    print("=" * 60)
    X_train, X_test, y_train, y_test, num_classes, class_names = cargar_y_preprocesar()
    print(f"\n  Clases  : {num_classes} → {class_names}")
    print(f"  Train   : {X_train.shape}")
    print(f"  Test    : {X_test.shape}\n")

    input_dim = X_train.shape[1]
    resultados = {}

    print("=" * 60)
    print("  PRUEBA 1: ANN (Red Neuronal Artificial)")
    print("=" * 60)
    ann = crear_ann(input_dim, num_classes)
    ann.summary()
    history_ann, ann = entrenar_modelo(ann, X_train, y_train, X_test, y_test, model_name="ANN", epochs=50, batch_size=64)
    resultados["ANN"] = evaluar_modelo(ann, X_test, y_test, history_ann, class_names, model_name="ANN")

    print("=" * 60)
    print("  PRUEBA 2: CNN (Red Neuronal Convolucional)")
    print("=" * 60)
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    cnn = crear_cnn(input_dim, num_classes)
    cnn.summary()
    history_cnn, cnn = entrenar_modelo(cnn, X_train_cnn, y_train, X_test_cnn, y_test, model_name="CNN", epochs=50, batch_size=64)
    resultados["CNN"] = evaluar_modelo(cnn, X_test_cnn, y_test, history_cnn, class_names, model_name="CNN")

    print("=" * 60)
    print("  PRUEBA 3: RNN-LSTM (Red Neuronal Recurrente)")
    print("=" * 60)
    WINDOW = 10
    X_train_rnn = X_train.reshape(X_train.shape[0], WINDOW, input_dim // WINDOW)
    X_test_rnn = X_test.reshape(X_test.shape[0], WINDOW, input_dim // WINDOW)
    rnn = crear_rnn(WINDOW, input_dim // WINDOW, num_classes)
    rnn.summary()
    history_rnn, rnn = entrenar_modelo(rnn, X_train_rnn, y_train, X_test_rnn, y_test, model_name="RNN", epochs=50, batch_size=64)
    resultados["RNN"] = evaluar_modelo(rnn, X_test_rnn, y_test, history_rnn, class_names, model_name="RNN")
    comparar_modelos(resultados)