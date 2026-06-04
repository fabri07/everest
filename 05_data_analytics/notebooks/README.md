# Notebooks de exploracion

Estos notebooks son el analisis exploratorio inicial (EDA) que se realizo antes del modelado. Documentan el proceso de descubrimiento y la comprension progresiva del dataset.

## Contenido

- `exploration/everest.ipynb` - Exploracion general del dataset, estructura de tablas y distribuciones basicas.
- `exploration/everest_expedition.ipynb` - Analisis a nivel expedicion: tendencias temporales, rutas, tasas de exito.
- `exploration/everest_members.ipynb` - Analisis a nivel miembro: demograficos, nacionalidades, roles.
- `exploration/everest_members_details.ipynb` - Analisis detallado: edad, ocupacion, relacion con outcomes.

## Nota

Estos notebooks usan las 4 categorias originales (cima_vive, cima_muere, abandona_vive, abandona_muere). El modelado final simplifica a un target binario (`exito`). Ver [`docs/CONCLUSIONES.md`](../docs/CONCLUSIONES.md) para los resultados definitivos.
