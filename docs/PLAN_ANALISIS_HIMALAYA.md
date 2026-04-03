# Plan De Analisis Himalaya

## Objetivo

Construir una base maestra a nivel montanista-expedicion para:

- describir los factores asociados a `cima_vive`, `cima_muere`, `abandona_vive` y `abandona_muere`
- preparar un dataset reproducible para modelado
- entrenar un baseline interpretable que sirva como punto de partida

## Regla de edad

- `age_raw = myear - yob`
- `age_clean` solo si `13 <= age_raw <= 85`
- si la edad queda fuera de rango:
  - no se usa como valor directo
  - se registra en `age_inconsistencies.csv`
  - se crea `age_out_of_range_flag`
- si falta `yob`:
  - `age_missing_flag = True`
  - `age_clean` queda nulo y luego se imputa en el pipeline del modelo

## Profesion y ocupacion

Se conservan dos columnas:

- `occupation_raw_clean`: texto limpio original
- `occupation_group`: agrupacion analitica

Grupos incluidos:

- `guide_instructor`
- `climber_alpinist`
- `doctor_health`
- `engineer`
- `military_police`
- `student`
- `business_exec`
- `scientist_academic`
- `media_photo`
- `trades_manual`
- `other`
- `unknown`

## Merge

- `members.csv` es la tabla base
- `exped.csv` se consolida previamente por `expid`
- si un `expid` aparece duplicado en `exped.csv`, se marca con `exped_ambiguous_expid`
- `peaks.csv` se une por `peakid`
- `refer.csv` se resume por `expid`

## Variables historicas

- `peak_prev_members`, `peak_prev_summits`, `peak_prev_deaths`
  - se calculan a nivel `peakid-expid-myear`
  - cada integrante de una misma expedicion recibe el mismo historial previo
  - solo usan informacion de expediciones anteriores sobre ese pico
- `peak_prev_summit_rate = peak_prev_summits / peak_prev_members`
- `peak_prev_death_rate = peak_prev_deaths / peak_prev_members`

## Variables eliminadas del modelo principal

Se quitan variables por alguno de estos motivos:

- fuga temporal o post-evento
- identificadores sin valor predictivo
- texto libre de baja calidad
- alta nulidad con poco valor en v1

Ejemplos:

- miembros:
  - nombres, ids internos, fechas de cima, detalles de muerte, rutas realizadas, uso real de oxigeno
- expediciones:
  - fechas de cima, razones de terminacion, muertes observadas, tiempos, accidentes, exito observado de la ruta
- picos:
  - duplicados de altura, texto libre de primera ascension, variables muy vacias

## Salidas del pipeline

- `data/processed/himalaya_master_clean.csv`
- `data/processed/himalaya_model_ready.csv`
- `reports/tables/age_inconsistencies.csv`
- `reports/tables/occupation_group_summary.csv`
- `reports/tables/occupation_raw_summary.csv`
- `reports/tables/region_summary.csv`
- `reports/metrics/data_quality_summary.json`
- `reports/metrics/baseline_metrics.json`
- `reports/predictions/baseline_predictions.csv`

## Estrategia de modelado

Baseline jerarquico:

1. modelo de `P(cima)`
2. modelo de `P(muerte | cima)`
3. modelo de `P(muerte | no cima)`

Luego se combinan las probabilidades para estimar:

- `P(cima_vive)`
- `P(cima_muere)`
- `P(abandona_vive)`
- `P(abandona_muere)`

## Validacion

Split temporal:

- train: `myear <= 2014`
- valid: `2015 <= myear <= 2019`
- test: `myear >= 2020`

Metricas:

- `balanced_accuracy`
- `f1`
- `macro_f1`
- `PR-AUC`

## Siguiente mejora recomendada

Una vez validado este baseline:

- reemplazar regresion logistica por CatBoost o LightGBM
- agregar calibration
- comparar con un multiclass directo
- crear SHAP global y local

## Modelo avanzado implementado

Script:

- `src/everest_analytics/train_catboost.py`

Salidas:

- `reports/metrics/catboost_metrics.json`
- `reports/predictions/catboost_predictions.csv`
- `reports/tables/catboost_calibration_summary.csv`
- `reports/figures/shap/*.csv`
- `reports/figures/shap/*.png`

Notas:

- el modelo usa CatBoost con variables numericas y categoricas
- la calibracion usa Platt scaling ajustado sobre el split `valid`
- las metricas de validacion se reportan sin calibrar
- la calibracion se evalua solo en `test` para evitar optimismo al medir sobre el mismo split usado para ajustarla
- SHAP explica el modelo CatBoost base; la calibracion se reporta por separado
