# Everest

Proyecto de analitica reproducible sobre la base Himalayan Database, organizado para que sirva como plantilla en datasets tabulares similares.

## Estructura

- `data/raw/`: datos fuente sin modificar.
- `data/processed/`: datasets limpios y listos para modelado.
- `src/everest_analytics/`: logica reutilizable de preparacion y modelado.
- `scripts/`: puntos de entrada para ejecutar el pipeline.
- `reports/metrics/`: metricas JSON.
- `reports/predictions/`: predicciones de validacion y test.
- `reports/tables/`: tablas derivadas para analisis y control de calidad.
- `reports/figures/`: figuras y SHAP.
- `notebooks/`: exploracion y notebooks de conclusiones.
- `docs/`: plan y arquitectura para replicar el enfoque.

## Flujo recomendado

1. Copiar los CSV originales en `data/raw/`.
2. Ejecutar `python scripts/prepare_data.py`.
3. Ejecutar `python scripts/train_baseline.py`.
4. Ejecutar `python scripts/train_catboost.py`.
5. Revisar resultados en `reports/`.

## Limpieza aplicada

- Se eliminaron caches, zips redundantes y experimentos auxiliares.
- Se retiraron modelos con resultados debiles o poco reutilizables como el multiclase de causas de no-cima y el dashboard `preintento_3c`.
- Se mantuvo el pipeline principal de preparacion, baseline y CatBoost jerarquico porque es el bloque mas solido para reutilizar.

## Patron reusable

Primero estabiliza la capa de preparacion de datos. Despues define el split temporal o de negocio. Recién ahi compara baseline interpretable contra un modelo mas potente. La carpeta `reports/` debe permitir revisar el proyecto sin depender del notebook.

