# Atajos de desarrollo. `make help` lista los objetivos.
PY      ?= python
JOBS    ?= $(shell expr $$(nproc 2>/dev/null || echo 2) - 1)
MODELS  ?= $(HOME)/Project_UAV/uav_v2/_03_cnn

.PHONY: help install install-train test test-all lint smoke baseline scenarios models docker gazebo clean

help:            ## esta ayuda
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:         ## paquete editable + inferencia int8 + herramientas de desarrollo
	$(PY) -m pip install -e ".[tflite,dev]"

install-train:   ## además TensorFlow (entrenamiento; ver docs/gpu.md)
	$(PY) -m pip install -e ".[tflite,train,dev]"

test:            ## tests rápidos (~15 s)
	$(PY) -m pytest -m "not slow" -q

test-all:        ## todos los tests (incl. episodio en lazo cerrado y CNN real si hay modelo)
	$(PY) -m pytest -q

lint:            ## ruff
	ruff check .

models:          ## importar modelos entrenados desde MODELS=<carpeta>
	nanouav models import $(MODELS) && nanouav models check

smoke:           ## suite de humo (2–4 min)
	nanouav suite configs/suites/smoke.yaml -j $(JOBS)

baseline:        ## línea base del paper C2 (horas; reanudable)
	nanouav suite configs/suites/baseline.yaml -j $(JOBS) --resume

scenarios:       ## escenarios de Gazebo offline, MH vs carril
	nanouav suite configs/suites/scenarios.yaml -j $(JOBS) --resume

docker:          ## imágenes de Gazebo (base + proyecto)
	docker build -f docker/Dockerfile.base -t nano_uav_sim:fixed .
	docker build -f docker/Dockerfile.sim  -t nano_uav_sim:v12 .

gazebo:          ## un episodio en Gazebo (WORLD=, SEED=, PLANNER=, ...)
	bash gazebo/gz_docker.sh

clean:           ## cachés de Python (no toca models/ data/ results/)
	find . -name __pycache__ -type d -prune -exec rm -rf {} + ; rm -rf .pytest_cache .ruff_cache *.egg-info src/*.egg-info
