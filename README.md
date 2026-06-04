# Everest

Proyecto de analitica reproducible sobre la base Himalayan Database, organizado como plantilla profesional para datasets tabulares similares.

## Objetivo

Clasificacion binaria: predecir si un miembro de expedicion **llega a la cima y sobrevive** (`exito=1`) versus cualquier otro resultado (`exito=0`: no hacer cima, morir en el intento, o ambas). Se entrenan tres modelos (CatBoost, Logistic Regression L1, Random Forest) y se cruzan sus feature importances para identificar las variables que realmente influyen.

## Arquitectura Por Etapas

- `01_data_ingestion/raw/`: fuentes originales sin modificar.
- `02_data_processing/data/`: datasets limpios y listos para modelado.
- `02_data_processing/scripts/`: ejecucion de limpieza y feature engineering.
- `03_model_training/scripts/`: entrenamiento del modelo unificado.
- `04_model_evaluation/metrics/`: metricas JSON de validacion y test.
- `04_model_evaluation/predictions/`: predicciones generadas por los modelos.
- `04_model_evaluation/figures/`: SHAP, consenso y figuras de evaluacion.
- `05_data_analytics/tables/`: tablas analiticas, importancias y ranking consensuado.
- `05_data_analytics/notebooks/`: exploracion y notebooks de conclusiones.
- `05_data_analytics/docs/`: plan y documentacion metodologica.
- `src/everest_analytics/`: logica reutilizable compartida entre etapas.

## Flujo Recomendado

1. Copiar los CSV originales en `01_data_ingestion/raw/`.
2. Ejecutar `python 02_data_processing/scripts/prepare_data.py`.
3. Ejecutar `python 03_model_training/scripts/train_success.py`.
4. Revisar resultados en `04_model_evaluation/` y `05_data_analytics/`.

## Modelos

- **CatBoost**: gradient boosting con soporte nativo de categoricas. SHAP para interpretabilidad.
- **Logistic Regression (L1/Lasso)**: coeficientes interpretables, seleccion automatica de features.
- **Random Forest**: importancia por impureza como tercer criterio independiente.
- **Consenso**: ranking cruzado de los 3 modelos para identificar features robustas.

## Patron Reutilizable

Primero estabiliza la etapa de procesamiento. Despues define el split temporal. Entrena multiples modelos con la misma target binaria y cruza importancias para separar senial real de ruido. Esta arquitectura separa evidencia tecnica de analitica para que otro proyecto similar pueda copiar la misma secuencia.
