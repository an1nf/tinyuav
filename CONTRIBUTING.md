# Cómo contribuir

## Entorno

```bash
pip install -e ".[tflite,dev]"
make test lint          # o: pytest -m "not slow" && ruff check .
```

## Convenciones

* **Idioma**: código, identificadores, comentarios, docstrings y mensajes en inglés; documentación
  (`README.md`, `docs/`, `CHANGELOG.md`) en español. Los datos que leen otros programas (nombres de
  escenarios, claves de JSON ya publicados, la línea `=== EPISODIO:` de Gazebo) no se traducen.
* **Imports** siempre desde el paquete (`from nanouav.planning.mh_planner import MHPlanner`); nunca
  `sys.path.insert` dentro de `src/`.
* **Rutas y banderas**: `nanouav.paths` y `nanouav.settings`, nunca rutas relativas al archivo ni
  `os.environ` directo.
* **Experimentos**: un módulo en `src/nanouav/experiments/`, registrado en su `__init__.py`, con
  semillas explícitas y nombres de salida que incluyan los parámetros (ver `docs/experimentos.md`).
* **Variantes de algoritmo** detrás de un atributo apagado por defecto, para no cambiar resultados
  publicados sin querer.
* Estilo: líneas ≤ 120 caracteres; `ruff check .` sin errores.

## Cambios que alteran resultados

`tests/test_regression.py` falla si cambia cualquier salida numérica de referencia. Si el cambio es
intencionado:

1. regenerar la referencia: `python tests/test_regression.py --update`;
2. explicar en `CHANGELOG.md` qué cambia y por qué;
3. volver a correr las suites afectadas y actualizar las tablas del informe en `docs/research/`.

## Ramas y commits

* `main` siempre pasa CI. Trabajo en ramas (`feat/…`, `fix/…`, `exp/rNN-…`) con *pull request*.
* Un commit = un cambio con sentido propio; mensaje en imperativo (inglés o español, pero
  consistente dentro de la rama).
* No se versionan modelos, datasets ni resultados (`models/`, `data/`, `results/` están en
  `.gitignore`); las cifras que importan se copian a `docs/research/` con el comando y el commit.
