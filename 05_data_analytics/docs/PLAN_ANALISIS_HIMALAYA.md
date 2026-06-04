# Plan De Analisis Himalaya

## Objetivo

Construir una base maestra a nivel montanista-expedicion para:

- identificar las variables que realmente influyen en **llegar a la cima y sobrevivir**
- preparar un dataset reproducible para modelado binario
- entrenar multiples modelos y cruzar sus importancias para separar senial real de ruido

## Target binario

El dataset original tiene fuerte desbalanceo en muerte (~1.3%), lo que impide modelar las 4 categorias por separado con confianza. Se simplifica a:

- `exito = 1`: hizo cima **y** sobrevivio (cima_vive)
- `exito = 0`: cualquier otro resultado (abandona_vive, abandona_muere, cima_muere)

Distribucion resultante: 41% exito vs 59% no-exito. Practicamente balanceado.

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

Grupos: `guide_instructor`, `climber_alpinist`, `doctor_health`, `engineer`, `military_police`, `student`, `business_exec`, `scientist_academic`, `media_photo`, `trades_manual`, `other`, `unknown`.

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

## Variables eliminadas del modelo

Se quitan variables por:

- fuga temporal o post-evento (fechas de cima, detalles de muerte, uso real de oxigeno)
- identificadores sin valor predictivo (nombres, ids internos)
- texto libre de baja calidad
- alta nulidad con poco valor

## Tratamiento de datos faltantes

- **Edad**: fuera de rango se convierte en NaN + flag; faltante se marca con flag
- **Ocupacion**: 34.5% faltante, se marca con flag y grupo "unknown"
- **Numericas**: CatBoost maneja NaN nativamente; LogReg/RF imputan con mediana
- **Categoricas**: NaN se rellena con "Unknown" como string
- **No se eliminan filas** por datos faltantes (excepto sin target msuccess/death)

## Separacion de features por cardinalidad

- **Baja cardinalidad** (todos los modelos): mseason, sex, occupation_group, peak_himal, peak_region, flags binarios, etc.
- **Alta cardinalidad** (solo CatBoost): citizen_clean (253 valores), status (554), exp_route1 (1265), exp_nation (100)

CatBoost maneja categoricas de alta cardinalidad nativamente. LogReg y RF usan OHE, que con 1000+ categorias genera features ruidosas y lentitud.

## Estrategia de modelado

Tres modelos independientes sobre el mismo target binario `exito`:

1. **CatBoost**: gradient boosting con soporte nativo de categoricas. 500 iteraciones, depth 6, early stopping.
2. **Logistic Regression (L1)**: coeficientes sparse interpretables. Solver liblinear.
3. **Random Forest**: 200 arboles, max_depth 12.

Luego se cruzan las importancias de los 3 modelos en un ranking consensuado.

## Validacion

Split temporal:

- train: `myear <= 2014` (64,076 filas)
- valid: `2015 <= myear <= 2019` (14,005 filas)
- test: `myear >= 2020` (10,919 filas)

Metricas: `balanced_accuracy`, `ROC AUC`, `PR AUC`, `F1`, `Brier score`, `log_loss`.

## Salidas del pipeline

- `02_data_processing/data/himalaya_model_ready.csv`
- `04_model_evaluation/metrics/success_metrics.json`
- `04_model_evaluation/metrics/data_quality_summary.json`
- `04_model_evaluation/predictions/success_predictions.csv`
- `04_model_evaluation/figures/consensus_feature_importance.png`
- `04_model_evaluation/figures/shap/exito_shap_*.png`
- `05_data_analytics/tables/consensus_feature_ranking.csv`
- `05_data_analytics/tables/importance_catboost.csv`
- `05_data_analytics/tables/importance_logreg.csv`
- `05_data_analytics/tables/importance_rf.csv`
- `05_data_analytics/tables/permutation_importance_catboost.csv`
