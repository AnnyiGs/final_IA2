import os
import time
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)


def entrenar_modelo(model, X_train, y_train, X_val, y_val,
                    model_name: str = "model",
                    epochs: int = 50,
                    batch_size: int = 64):
    """
    Parámetros
    ----------
    model       : modelo Keras compilado
    X_train     : datos de entrenamiento
    y_train     : etiquetas one-hot de entrenamiento
    X_val       : datos de validación
    y_val       : etiquetas one-hot de validación
    model_name  : nombre del modelo (para guardar pesos)
    epochs      : número máximo de épocas
    batch_size  : tamaño de lote

    Retorna
    -------
    history, model  (historial de entrenamiento y modelo entrenado)
    """

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/{model_name}_best.keras"

    callbacks = [
        EarlyStopping(monitor="val_accuracy",
                      patience=10,
                      restore_best_weights=True,
                      verbose=1),

        ReduceLROnPlateau(monitor="val_loss",
                          factor=0.5,
                          patience=5,
                          min_lr=1e-6,
                          verbose=1),

        ModelCheckpoint(filepath=checkpoint_path,
                        monitor="val_accuracy",
                        save_best_only=True,
                        verbose=0)
    ]

    print(f"\n  Entrenando {model_name}  (max {epochs} épocas, "
          f"batch={batch_size})…\n")

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - t0

    best_acc = max(history.history["val_accuracy"])
    print(f"\n  ✔  {model_name} entrenado en {elapsed:.1f}s  "
          f"| Mejor val_accuracy = {best_acc:.4f}\n")

    # Guardar tiempo de entrenamiento en el historial para comparación
    history.history["training_time"] = elapsed

    return history, model
