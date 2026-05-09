# Proyecto de Clasificación de Ritmos Cardíacos con Redes Neuronales

Este proyecto implementa y compara tres tipos de redes neuronales (ANN, CNN y RNN-LSTM) para la clasificación de ritmos cardíacos utilizando el dataset MIT-BIH. A continuación, se describen los pasos principales y los resultados obtenidos.

## Descripción del Proyecto

El objetivo principal es entrenar modelos de redes neuronales para clasificar señales de ECG en cinco categorías:
- **Normal (N)**
- **Supraventricular (S)**
- **Ventricular (V)**
- **Fusión (F)**
- **Desconocido (Q)**

Se utilizan tres arquitecturas de redes neuronales:
1. **ANN (Artificial Neural Network)**
2. **CNN (Convolutional Neural Network)**
3. **RNN-LSTM (Recurrent Neural Network con LSTM)**

## Resultados

### Comparación de Modelos

La siguiente gráfica muestra la comparación de las métricas de los tres modelos:

![Comparación de Modelos](resultados/comparacion_modelos.png)

### Resultados por Modelo

#### ANN (Red Neuronal Artificial)
- **Matriz de Confusión:**

![Matriz de Confusión ANN](resultados/ANN_confusion_matrix.png)

- **Evolución del Entrenamiento:**

![Evolución del Entrenamiento ANN](resultados/ANN_history.png)

#### CNN (Red Neuronal Convolucional)
- **Matriz de Confusión:**

![Matriz de Confusión CNN](resultados/CNN_confusion_matrix.png)

- **Evolución del Entrenamiento:**

![Evolución del Entrenamiento CNN](resultados/CNN_history.png)

#### RNN-LSTM (Red Neuronal Recurrente)
- **Matriz de Confusión:**

![Matriz de Confusión RNN](resultados/RNN_confusion_matrix.png)

- **Evolución del Entrenamiento:**

![Evolución del Entrenamiento RNN](resultados/RNN_history.png)

## Requisitos

- Python 3.13
- Librerías necesarias:
  - `tensorflow`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

## Ejecución

1. Clonar el repositorio o descargar los archivos.
2. Instalar las dependencias necesarias.
3. Ejecutar el script principal:

```bash
python zProFinalSSPIA2.py
```

## Notas

- Los resultados se guardan automáticamente en la carpeta `resultados`.
- El modelo con mejor desempeño se destaca en la comparación final.

