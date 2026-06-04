# Conclusiones: Factores De Exito En Expediciones Al Himalaya

## Pregunta central

Que variables influyen realmente en que un montanista **llegue a la cima y sobreviva**?

## Datos

89,000 registros de miembros de expediciones al Himalaya (Himalayan Database). Se define un target binario:

- `exito = 1`: hizo cima y sobrevivio (36,669 casos, 41.2%)
- `exito = 0`: cualquier otro resultado (52,331 casos, 58.8%)

Esta simplificacion resuelve el desbalanceo extremo del dataset original, donde las muertes representaban solo el 1.3% del total y no habia suficientes datos para modelar las 4 categorias (cima_vive, cima_muere, abandona_vive, abandona_muere) de forma confiable.

## Rendimiento de los modelos (test set, datos >= 2020)

| Modelo | Balanced Acc | ROC AUC | PR AUC | Brier | F1 |
|---|---|---|---|---|---|
| CatBoost | 0.680 | 0.758 | 0.832 | 0.195 | 0.743 |
| Logistic Regression (L1) | 0.645 | 0.795 | 0.864 | 0.181 | 0.796 |
| Random Forest | 0.724 | 0.799 | 0.875 | 0.193 | 0.722 |

Los tres modelos muestran capacidad predictiva real (ROC AUC 0.76-0.80). Random Forest logra el mejor balance entre clases (balanced accuracy 0.724) y la mejor PR AUC (0.875). Logistic Regression tiene el menor Brier score (0.181), lo que indica probabilidades mejor calibradas.

Ningun modelo individual domina en todas las metricas, lo que refuerza la decision de cruzar importancias en vez de confiar en uno solo.

## Variables que realmente importan

### Top 9 por consenso (aparecen en el top 10 de 2 o 3 modelos)

| Rank | Variable | En top10 de | Interpretacion |
|---|---|---|---|
| 1 | **bconly** | 3/3 modelos | Si el miembro solo fue al campo base. Es la senial mas fuerte: quien no pasa del campo base, por definicion no hace cima. |
| 2 | **hired** | 2/3 | Si es personal contratado (sherpa, porteador). Los contratados tienen tasas de exito diferentes al resto. |
| 3 | **peak_prev_summit_rate** | 2/3 | Tasa historica de exito del pico. Picos con historial de exito alto predicen exito futuro. |
| 4 | **myear** | 2/3 | Ano de la expedicion. Las tasas de exito han mejorado con el tiempo (mejor equipamiento, rutas mas conocidas). |
| 5 | **nottobc** | 2/3 | Si el miembro no llego al campo base. Complemento de bconly. |
| 6 | **peak_himal** | 2/3 | Cadena montaniosa (Everest, Annapurna, Kanchenjunga, etc.). Cada cadena tiene su propia dificultad y tasas de exito. |
| 7 | **exp_camps** | 2/3 | Numero de campamentos de la expedicion. Mas campamentos implica mejor logistica. |
| 8 | **exp_rope** | 2/3 | Uso de cuerdas fijas. Indicador de preparacion tecnica de la expedicion. |
| 9 | **peak_heightm** | 2/3 | Altura del pico. A mayor altitud, menor probabilidad de exito. |

### Variables con importancia moderada (top 10 en 1 modelo)

- **occupation_group**: la profesion influye (guias e instructores vs estudiantes vs ejecutivos)
- **sherpa**: si el miembro es sherpa
- **peak_phost**: pais anfitrion del pico
- **peak_prev_death_rate**: tasa historica de muerte del pico
- **peak_restrict**: si el pico tiene restricciones de acceso

### Variables con poca o nula importancia

- **sex**: el sexo no aparece como predictor fuerte en ningun modelo
- **deputy**, **disabled**, **tibetan**, **exp_parapente**: importancia cero en SHAP y permutation importance
- **age_clean**: la edad tiene importancia moderada en permutation importance (rank 3) pero no destaca en los rankings de modelo, sugiriendo que su efecto es real pero no dominante

## Validacion cruzada de importancias

Se usaron 4 metodos independientes:

1. **CatBoost feature importance** (ganancia en splits)
2. **LogReg L1 coeficientes** (magnitud absoluta con regularizacion)
3. **Random Forest impurity importance** (reduccion de impureza)
4. **Permutation importance** (caida en ROC AUC al permutar cada feature en test)

Las 4 perspectivas coinciden en que bconly, hired, exp_camps y peak_himal son consistentemente relevantes. Las variables que solo aparecen en 1 metodo (como exp_route1 en CatBoost o exp_rope en LogReg) podrian ser artefactos del algoritmo.

Un caso interesante: **peak_prev_summit_rate** aparece como top 1 en RF y top 7 en CatBoost, pero tiene permutation importance **negativa** (-0.0067). Esto significa que al permutar esa variable en test, el modelo mejora. La explicacion mas probable es data leakage parcial: la tasa historica de exito del pico esta altamente correlacionada con la identidad del pico (peak_himal), y al permutar una, la otra compensa. La senial es real, pero esta redundante con peak_himal.

## Limitaciones

- El dataset no registra variables fisiologicas (VO2 max, aclimatacion), psicologicas (experiencia previa) ni meteorologicas detalladas. Estas variables probablemente explican una porcion importante de la varianza no capturada.
- El split temporal (train <= 2014, test >= 2020) mide generalizacion real, pero el alpinismo cambio significativamente en esa decada (comercializacion, mejor equipamiento). Los modelos capturan el alpinismo contemporaneo solo parcialmente.
- Las categoricas de alta cardinalidad (citizen_clean, status, exp_route1) solo fueron evaluadas por CatBoost. Su importancia real podria estar subestimada en el consenso.

## Conclusion

Los factores mas determinantes para llegar a la cima y sobrevivir son:

1. **La logistica de la expedicion**: numero de campamentos, cuerdas fijas, personal contratado
2. **La dificultad intrinseca del pico**: altura, cadena montaniosa, historial de exito
3. **La epoca**: las expediciones recientes tienen mejores tasas de exito

Las variables individuales del montanista (edad, sexo, profesion) tienen un peso menor comparado con las variables de la expedicion y el pico. Esto sugiere que en el alpinismo himalayico, **la preparacion colectiva importa mas que las caracteristicas individuales**.
