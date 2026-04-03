# Arquitectura Analitica Reutilizable

## Capas

1. `data/raw`: datos originales, sin tocar.
2. `data/processed`: tablas limpias y enriquecidas.
3. `src/<proyecto>`: reglas de negocio, features, validacion y modelos.
4. `scripts`: ejecucion del flujo sin notebooks.
5. `reports`: metricas, predicciones, tablas y figuras.
6. `notebooks`: exploracion o comunicacion, no logica productiva.

## Secuencia

1. Ingesta y validacion de claves.
2. Consolidacion y resolucion de duplicados.
3. Quality checks exportables.
4. Baseline interpretable.
5. Modelo principal.
6. Reporte reproducible.

## Regla para futuros datasets

- Si un experimento no supera claramente al baseline o no mejora interpretabilidad, queda fuera del flujo principal.
- La logica operativa va en `src` y `scripts`.
- Los notebooks son apoyo, no dependencia.
