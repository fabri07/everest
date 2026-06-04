# Everest

Proyecto de analitica reproducible sobre la base [Himalayan Database](https://www.himalayandatabase.com/) (~89,000 registros de expediciones). Identifica las variables que realmente influyen en que un montanista **llegue a la cima y sobreviva**.

## Resultados principales

### Rendimiento de los modelos (test set, expediciones >= 2020)

| Modelo | Balanced Acc | ROC AUC | PR AUC | F1 |
|---|---|---|---|---|
| CatBoost | 0.680 | 0.758 | 0.832 | 0.743 |
| Logistic Regression (L1) | 0.645 | 0.795 | 0.864 | 0.796 |
| Random Forest | **0.724** | **0.799** | **0.875** | 0.722 |

### Top variables por consenso de 3 modelos

| Rank | Variable | Descripcion | En top10 de |
|---|---|---|---|
| 1 | bconly | Solo fue al campo base | 3/3 modelos |
| 2 | hired | Personal contratado (sherpa/porteador) | 2/3 |
| 3 | peak_prev_summit_rate | Tasa historica de exito del pico | 2/3 |
| 4 | myear | Ano de la expedicion | 2/3 |
| 5 | peak_himal | Cadena montanosa | 2/3 |
| 6 | exp_camps | Numero de campamentos | 2/3 |
| 7 | exp_rope | Uso de cuerdas fijas | 2/3 |
| 8 | peak_heightm | Altura del pico en metros | 2/3 |

**Conclusion**: la logistica de la expedicion (campamentos, cuerdas, personal) y la dificultad del pico (altura, historial) importan mas que las caracteristicas individuales del montanista (edad, sexo, profesion).

Analisis completo en [`05_data_analytics/docs/CONCLUSIONES.md`](05_data_analytics/docs/CONCLUSIONES.md).

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

Requiere Python >= 3.10.

## Flujo de ejecucion

```bash
# 1. Preparar datos (limpieza, merges, feature engineering)
python 02_data_processing/scripts/prepare_data.py

# 2. Entrenar modelos y generar outputs
python 03_model_training/scripts/train_success.py
```

Los scripts deben ejecutarse desde la raiz del proyecto.

## Arquitectura

```
01_data_ingestion/raw/          Fuentes originales (members, exped, peaks, refer)
02_data_processing/data/        Datasets limpios y model-ready
02_data_processing/scripts/     Pipeline de limpieza y feature engineering
03_model_training/scripts/      Entrenamiento unificado (CatBoost + LogReg + RF)
04_model_evaluation/metrics/    Metricas JSON de validacion y test
04_model_evaluation/predictions/ Predicciones de los 3 modelos
04_model_evaluation/figures/    SHAP plots y grafico de consenso
05_data_analytics/tables/       Importancias por modelo y ranking consensuado
05_data_analytics/notebooks/    Notebooks de exploracion
05_data_analytics/docs/         Plan, arquitectura y conclusiones
src/everest_analytics/          Logica reutilizable compartida entre etapas
```

## Modelos

- **CatBoost**: gradient boosting con soporte nativo de categoricas. SHAP para interpretabilidad.
- **Logistic Regression (L1/Lasso)**: coeficientes interpretables, seleccion automatica de features.
- **Random Forest**: importancia por impureza como tercer criterio independiente.
- **Consenso**: ranking cruzado de los 3 modelos para identificar features robustas.

## Metodologia

- **Target binario**: `exito = 1` (cima + sobrevive) vs `exito = 0` (todo lo demas). Split 41/59, practicamente balanceado.
- **Split temporal**: train <= 2014, validation 2015-2019, test >= 2020.
- **Feature importance cruzada**: se comparan importancias de 3 modelos independientes. Las variables que aparecen consistentemente en los 3 son senial real, no artefactos de un algoritmo.
- **Datos faltantes**: no se eliminan filas. Se imputan con mediana/moda y se crean flags para que el modelo aprenda de la ausencia.

Documentacion metodologica detallada en [`05_data_analytics/docs/PLAN_ANALISIS_HIMALAYA.md`](05_data_analytics/docs/PLAN_ANALISIS_HIMALAYA.md).
