# Entrenamiento con GPU

Solo el entrenamiento (`train`, `train-tartanair`, `r03`, `r11 train`) usa GPU. La generación de
datasets, el renderer, la simulación en lazo cerrado y la inferencia int8 corren en CPU (numpy +
TFLite). La CNN usa 1–2 GB de VRAM con batch 128, así que 8 GB sobran.

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: usar WSL2 (TF-GPU no corre nativo)
pip install -e ".[train,tflite,dev]"
pip install "tensorflow[and-cuda]"                     # trae CUDA/cuDNN como paquetes nvidia-* de pip
nanouav doctor --gpu                                   # debe listar la GPU
```

### Si TensorFlow no encuentra las bibliotecas de CUDA

Con las ruedas `nvidia-*` de pip, TensorFlow normalmente localiza las bibliotecas solo. Si
`doctor --gpu` muestra una lista vacía y el log menciona `libcudnn` o `libcublas`, añadir sus
carpetas a `LD_LIBRARY_PATH` antes de lanzar el entrenamiento:

```bash
export LD_LIBRARY_PATH="$(python - <<'EOF'
import glob, os, nvidia
print(':'.join(sorted(glob.glob(os.path.join(list(nvidia.__path__)[0], '*', 'lib')))))
EOF
):$LD_LIBRARY_PATH"
```

### GPU Pascal (TITAN X, GTX 10xx)

En la TITAN X (Pascal, *compute capability* 6.1) el entrenamiento solo funcionó con esta
combinación:

1. **cuDNN fijado** a la versión con la que funcionó:
   `pip install nvidia-cudnn-cu12==9.3.0.75`. Las versiones posteriores dieron errores de cuDNN en
   esta GPU.
2. **Sin XLA**: los modelos se compilan con `jit_compile=False` (en `train_cnn.py`,
   `r3_lnca_train.py` y `r11_vel_net.py`),
   porque el autotuner de XLA no encuentra kernels de cuDNN fusionados en Pascal.

En GPUs más recientes no debería hacer falta ninguna de las dos cosas.

## Lanzar

```bash
nanouav run train --variants XS,S,M,L --epochs 60 --n-train 30000 --batch 128 --lr 4e-3
nanouav run r03 --models M_ToF --init-from models/model_M.keras --install \
                --n-train 30000 --epochs 60 --batch 128 --lr 4e-3
```

En una suite, los trabajos de entrenamiento tienen que declarar `gpu: true` (el runner oculta la
GPU por defecto para que los trabajos de simulación en paralelo no reserven VRAM):

```yaml
jobs:
  - id: train-m-tof
    run: r03 --models M_ToF --init-from models/model_M.keras --install --epochs 60
    gpu: true
```
