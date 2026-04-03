# Arquitectura Analitica Reutilizable

## Etapas

1. `01_data_ingestion`
   Entrada de datos originales y diccionarios.
2. `02_data_processing`
   Limpieza, consolidacion, features y datasets listos para modelado.
3. `03_model_training`
   Entrenamiento de baseline y modelo principal.
4. `04_model_evaluation`
   Metricas, predicciones y explicabilidad del modelo.
5. `05_data_analytics`
   Tablas de negocio, notebooks y documentacion.

## Regla De Diseno

- Cada carpeta responde a una etapa del trabajo, no a un tipo de archivo aislado.
- La salida de una etapa alimenta la siguiente.
- Los notebooks quedan al final del flujo, no en el centro de la logica.
- La logica compartida se mantiene en `src/everest_analytics/`.

## Secuencia Profesional Recomendada

1. Ingesta y validacion de claves.
2. Procesamiento y feature engineering.
3. Entrenamiento reproducible.
4. Evaluacion con metricas y explicabilidad.
5. Analitica, storytelling y documentacion.
