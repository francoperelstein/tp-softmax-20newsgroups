# TP - Clasificación Multiclase de Texto con Regresión Softmax

Trabajo práctico de la materia **Redes Neuronales en Bioingeniería** (ITBA). Implementa un clasificador
de texto multiclase (Regresión Softmax en PyTorch) sobre el dataset **20 Newsgroups**
(~18.000 posts de foros de Usenet en 20 categorías), con un pipeline de preprocesamiento
configurable, vectorización TF-IDF, búsqueda de hiperparámetros con Optuna, y tracking
de experimentos con MLflow y TensorBoard.

Consigna base: [TP-Regresion-Softmax-TNG](https://github.com/cselmo/TP-Regresion-Softmax-TNG)

## Cómo correrlo

1. Instalar las dependencias con `pip install -r requirements.txt` (en Google Colab alcanza con la primera celda del notebook, que instala `mlflow` y `optuna`; el resto ya viene instalado).
2. Abrir `TP_Regresion_Softmax_TNG.ipynb` en [Google Colab](https://colab.research.google.com/) con GPU y ejecutar las celdas en orden desde arriba (*Entorno de ejecución > Ejecutar todas*).
3. Los pasos que más tardan son la celda de precálculo de preprocesamiento (sección 6) y las dos búsquedas de Optuna (secciones 6 y 10, 15 trials cada una).

## Estructura del notebook

1. Datos: carga de 20 Newsgroups.
2. Preprocesamiento configurable (stopwords, stemming/lematización).
3. Vectorización TF-IDF.
4. Modelo: Regresión Softmax (PyTorch).
5. Entrenamiento.
6. Búsqueda de hiperparámetros con Optuna.
7. Entrenamiento final.
8. Evaluación final.
9. Preguntas de reflexión.
10. Análisis extendido: resultados de la búsqueda, dinámica de entrenamientos y tiempos de convergencia.
11. Análisis controlado: baselines, un factor por vez con 5 semillas, diseño factorial lr × batch, convergencia sistemática y early stopping con test evaluado una sola vez al final.

## Resultados principales

- Mejor combinación encontrada por Optuna: sacando stopwords, stemming, unigramas, `max_features=10000` y `min_df=2`.
- Accuracy en test (protocolo limpio, sección 11.6): 0.6619 ± 0.0017 (5 semillas; entrenado en todo el train durante 15 épocas, elegidas por early stopping sobre una partición de validación del train).
- Accuracy en test del entrenamiento final original (sección 8): ~0.6569 (20 épocas). En esa corrida las curvas "val" se calcularon sobre el test, aunque no se usaron para decidir nada (las épocas estaban fijas de antemano). La diferencia entre ambos números (+0.5 puntos) está dentro del error estándar del test (±0.005), por lo que no se afirma una mejora.
- Las categorías con más confusión son las de religión (`talk.religion.misc`, `alt.atheism`), política (`talk.politics.*`) y los subgrupos de `comp.`, detallado en la sección 9 (pregunta 9) y 10 del notebook.


## Hallazgos de la sección 11 (validación, 5 semillas, configuración base = modelo final)

- Baselines: clase más frecuente 0.053, azar 0.047 y Complement Naive Bayes sin ajustar 0.7384, prácticamente igual al softmax de la búsqueda (0.7415): la diferencia es menor que el error estándar de la validación (±0.009).
- Hiperparámetros (un factor por vez): `max_features` es el de mayor efecto (2000 → −8.5 puntos); el `lr` tiene una meseta entre 0.0069 y 0.0098; el `batch_size` no cambia la accuracy (≤ 0.3 puntos), solo el tiempo por época.
- Trial 7 (bajo rendimiento): se debe principalmente al lr bajo (+3.3 puntos al subirlo de 0.0015 a 0.0069), no al batch (−0.6 puntos); con más épocas el lr bajo alcanza el mismo val_acc, es decir, estaba subentrenado en 5 épocas.
- Early stopping: la val_loss mínima ocurre en la época 15.4 ± 0.5; la accuracy llega a su máximo antes (época 6–7), por lo que parar por val_loss es el criterio estándar pero deja ≈ 1 punto de accuracy de validación frente a parar por val_acc.

## Limitaciones y reproducibilidad

- Las búsquedas de Optuna de las secciones 6 y 10 no tienen semilla en el sampler y no son reproducibles bit a bit. La sección 10 analiza una segunda búsqueda (`study_v2`) porque la primera no guardaba el historial por época ni la duración de cada trial.
- Los experimentos de la sección 11 usan semillas fijas (0 a 4). El desvío reportado es entre semillas (inicialización y orden de los batches) y no incluye la variabilidad de la partición train/val.
- La partición de validación es una sola (80/20 del train, estratificada, `SEED = 42`).

## Aclaración sobre la corrida del pipeline

Las secciones 1 a 10 se ejecutaron en Colab con GPU en la corrida original y la sección 11 en una corrida posterior, sin repetir las búsquedas de Optuna.

## Declaración de uso de IA

Usé Claude (Anthropic) para entender el código del notebook, razonar los resultados y llegar a conclusiones, y como base para redactar algunos de los análisis, que luego reescribí con mis palabras y verifiqué contra los outputs.
