# Arquitectura Analitica Reutilizable

## Etapas

1. `01_data_ingestion`
   Entrada de datos originales y diccionarios.
2. `02_data_processing`
   Limpieza, consolidacion, features y datasets listos para modelado.
3. `03_model_training`
   Entrenamiento unificado: CatBoost, Logistic Regression y Random Forest sobre target binario.
4. `04_model_evaluation`
   Metricas, predicciones, SHAP y grafico de consenso.
5. `05_data_analytics`
   Tablas de importancia, ranking consensuado, notebooks y documentacion.

## Regla De Diseno

- Cada carpeta responde a una etapa del trabajo, no a un tipo de archivo aislado.
- La salida de una etapa alimenta la siguiente.
- Los notebooks quedan al final del flujo, no en el centro de la logica.
- La logica compartida se mantiene en `src/everest_analytics/`.

## Secuencia Profesional Recomendada

1. Ingesta y validacion de claves.
2. Procesamiento y feature engineering.
3. Entrenamiento reproducible con multiples modelos.
4. Evaluacion con metricas, explicabilidad (SHAP) y consenso cruzado.
5. Analitica, storytelling y documentacion.

## Patron De Consenso

Para responder "que variables realmente importan" se cruzan tres fuentes independientes:

- CatBoost feature importance (ganancia en splits)
- Logistic Regression L1 (magnitud absoluta de coeficientes)
- Random Forest (importancia por impureza)

Las features que aparecen en el top 10 de los 3 modelos son las mas robustas. Las que solo aparecen en un modelo pueden ser artefactos del algoritmo.
