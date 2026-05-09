import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, recall_score, precision_score
)

os.makedirs("resultados", exist_ok=True)

# ── Paleta de colores por modelo ──────────────────────────────────
COLORS = {"ANN": "#4C72B0", "CNN": "#DD8452", "RNN": "#55A868", "RNN_LSTM": "#55A868"}


# ─────────────────────────────────────────────────────────────────
def evaluar_modelo(model, X_test, y_test, history,
                   class_names: list, model_name: str = "model") -> dict:
    """
    Genera:
      1. Reporte de clasificación (precision, recall, F1)
      2. Matriz de confusión
      3. Curvas de accuracy y loss
    Retorna diccionario con métricas numéricas para comparación.
    """

    # ── Predicciones ──────────────────────────────────────────────
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test,      axis=1)

    # ── Métricas escalares ────────────────────────────────────────
    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_true, y_pred,    average="weighted", zero_division=0)
    f1        = f1_score(y_true, y_pred,         average="weighted", zero_division=0)
    val_loss  = min(history.history["val_loss"])
    t_time    = history.history.get("training_time", 0)

    print(f"\n{'─'*55}")
    print(f"  Resultados — {model_name}")
    print(f"{'─'*55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  Val Loss  : {val_loss:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=class_names, zero_division=0)}")

    # ── 1. Matriz de confusión ────────────────────────────────────
    _plot_confusion_matrix(y_true, y_pred, class_names, model_name)

    # ── 2. Curvas accuracy & loss ─────────────────────────────────
    _plot_history(history, model_name)

    return {
        "accuracy":  acc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "val_loss":  val_loss,
        "time":      t_time
    }


# ─────────────────────────────────────────────────────────────────
def _plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names)-1)))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)

    ax.set_title(f"Matriz de Confusión — {model_name}", fontsize=14, pad=12)
    ax.set_xlabel("Predicción", fontsize=11)
    ax.set_ylabel("Real",       fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = f"resultados/{model_name}_confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  ✔  Guardado: {path}")


# ─────────────────────────────────────────────────────────────────
def _plot_history(history, model_name):
    hist  = history.history
    color = COLORS.get(model_name, "#333333")
    epochs = range(1, len(hist["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Evolución del Entrenamiento — {model_name}", fontsize=13)

    # Accuracy
    axes[0].plot(epochs, hist["accuracy"],     color=color,   label="Train Accuracy")
    axes[0].plot(epochs, hist["val_accuracy"], color=color,   label="Val Accuracy",
                 linestyle="--", alpha=0.8)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(epochs, hist["loss"],     color=color,   label="Train Loss")
    axes[1].plot(epochs, hist["val_loss"], color=color,   label="Val Loss",
                 linestyle="--", alpha=0.8)
    axes[1].set_title("Pérdida (Loss)")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = f"resultados/{model_name}_history.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  ✔  Guardado: {path}")


# ─────────────────────────────────────────────────────────────────
def comparar_modelos(resultados: dict):
    """
    Genera tabla y gráfico comparativo de ANN, CNN y RNN.
    """
    modelos = list(resultados.keys())
    metricas = ["accuracy", "precision", "recall", "f1", "val_loss", "time"]
    labels   = ["Accuracy", "Precision", "Recall", "F1-score", "Val Loss", "Tiempo (s)"]

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
    print()

    # ── Gráfico de barras comparativo ─────────────────────────────
    metricas_plot = ["accuracy", "precision", "recall", "f1"]
    labels_plot   = ["Accuracy", "Precision", "Recall", "F1-score"]

    x    = np.arange(len(metricas_plot))
    ancho = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, modelo in enumerate(modelos):
        vals = [resultados[modelo][m] for m in metricas_plot]
        bars = ax.bar(x + i * ancho, vals, ancho,
                      label=modelo,
                      color=list(COLORS.values())[i % len(COLORS)])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + ancho)
    ax.set_xticklabels(labels_plot)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Puntuación")
    ax.set_title("Comparación de Métricas: ANN vs CNN vs RNN")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = "resultados/comparacion_modelos.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  ✔  Guardado: {path}")

    # ── Mejor modelo ──────────────────────────────────────────────
    mejor = max(resultados, key=lambda m: resultados[m]["accuracy"])
    print(f"\n  🏆  Mejor modelo por Accuracy: {mejor} "
          f"({resultados[mejor]['accuracy']:.4f})\n")
