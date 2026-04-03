# Everest

Proyecto de analitica reproducible sobre la base Himalayan Database, organizado como plantilla profesional para datasets tabulares similares.

## Arquitectura Por Etapas

- `01_data_ingestion/raw/`: fuentes originales sin modificar.
- `02_data_processing/data/`: datasets limpios y listos para modelado.
- `02_data_processing/scripts/`: ejecucion de limpieza y feature engineering.
- `03_model_training/scripts/`: entrenamiento de baseline y modelo principal.
- `04_model_evaluation/metrics/`: metricas JSON de validacion y test.
- `04_model_evaluation/predictions/`: predicciones generadas por los modelos.
- `04_model_evaluation/figures/`: SHAP y figuras de evaluacion.
- `05_data_analytics/tables/`: tablas analiticas y resumenes de negocio.
- `05_data_analytics/notebooks/`: exploracion y notebooks de conclusiones.
- `05_data_analytics/docs/`: plan y documentacion metodologica.
- `src/everest_analytics/`: logica reutilizable compartida entre etapas.

## Flujo Recomendado

1. Copiar los CSV originales en `01_data_ingestion/raw/`.
2. Ejecutar `python 02_data_processing/scripts/prepare_data.py`.
3. Ejecutar `python 03_model_training/scripts/train_baseline.py`.
4. Ejecutar `python 03_model_training/scripts/train_catboost.py`.
5. Revisar resultados en `04_model_evaluation/` y `05_data_analytics/`.

## Criterios De Limpieza

- Se eliminaron caches, zips redundantes y experimentos auxiliares.
- Se retiraron modelos con resultados debiles o poco reutilizables como el multiclase de causas de no-cima y el dashboard `preintento_3c`.
- Se mantuvo el pipeline principal de preparacion, baseline y CatBoost jerarquico porque es el bloque mas solido para reutilizar.

## Patron Reutilizable

Primero estabiliza la etapa de procesamiento. Despues define el split temporal o de negocio. Recién ahi compara baseline interpretable contra un modelo mas potente. Esta arquitectura separa evidencia tecnica de analitica para que otro proyecto similar pueda copiar la misma secuencia sin mezclar archivos.
