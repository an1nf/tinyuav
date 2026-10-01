#!/bin/bash
# Lanza el contenedor de Gazebo desde el HOST (raíz del proyecto), con la GPU disponible para el render.
#
#   bash gazebo/gz_docker.sh                         → simulación con ventanas (run_sim.sh)
#   WORLD=bosque SEED=2 bash gazebo/gz_docker.sh     → cualquier variable de run_sim.sh se pasa al contenedor
#   bash gazebo/gz_docker.sh diag                    → diagnóstico de GPU/EGL dentro del contenedor
#   bash gazebo/gz_docker.sh bash                    → shell interactiva
#   DOCKER="sudo docker" bash gazebo/gz_docker.sh    → si tu usuario no está en el grupo docker
#   NO_GPU=1 ...                                     → render por software (lento)
#
# Por qué: con solo "--gpus all" el contenedor recibe las capacidades compute,utility (CUDA) pero NO
# graphics → no hay libEGL_nvidia/libGLX_nvidia → Mesa intenta dri2 ("libEGL warning: failed to create
# dri2 screen") y cae a llvmpipe: arranque de ~40–70 s y factor de tiempo real bajo.
# NVIDIA_DRIVER_CAPABILITIES=all monta las bibliotecas gráficas del driver del host (nvidia-container-toolkit).
IMG=${IMG:-nano_uav_sim:v12}
DOCKER=${DOCKER:-docker}
ROOTDIR=$(cd "$(dirname "$0")/.." && pwd)

GPU_ARGS="--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all"
[ "${NO_GPU:-0}" = "1" ] && GPU_ARGS="-e LIBGL_ALWAYS_SOFTWARE=1"
X_ARGS=""
if [ -n "$DISPLAY" ] && [ "${HEADLESS:-false}" != "true" ]; then
    xhost +local:docker >/dev/null 2>&1 || true
    X_ARGS="-e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -e XDG_RUNTIME_DIR=/tmp/xdg -v /tmp/.X11-unix:/tmp/.X11-unix"
fi
ENVS=""
for v in WORLD SEED WORLD_KW REGEN VARIANT PROFILE PLANNER CONSISTENT RF_DYN V_CRUISE GOAL GOAL_X GOAL_Y LOOP \
         T_MAX EXIT_ON_DONE HEADLESS RUN_TAG CONTROL_MODE YAW_SIGN MH_GRID_RES; do
    [ -n "${!v}" ] && ENVS="$ENVS -e $v=${!v}"
done
TTY=""; [ -t 0 ] && TTY="-it"

if [ "$1" = "diag" ]; then
    CMD='echo "== nvidia-smi"; nvidia-smi -L || echo "SIN GPU: ¿nvidia-container-toolkit instalado? ¿--gpus all?";
echo "== vendors EGL (debe aparecer 10_nvidia.json)"; ls /usr/share/glvnd/egl_vendor.d/ 2>/dev/null;
echo "== libEGL_nvidia"; ldconfig -p | grep -E "libEGL_nvidia|libGLX_nvidia" || echo "NO: falta capacidad graphics";
echo "== prueba de render sin ventana (10 s)"; source /opt/ros/jazzy/setup.bash;
timeout 25 gz sim -s -r --headless-rendering --iterations 2500 /sim/nano_uav/gazebo/worlds/nano_uav_world.sdf 2>&1 | grep -iE "egl|ogre|render|error" | head -8;
echo "(si no hay errores de EGL, el render usa la GPU)"'
    exec $DOCKER run --rm $TTY $GPU_ARGS -v "$ROOTDIR":/sim/nano_uav -w /sim/nano_uav $IMG bash -c "$CMD"
fi
if [ $# -gt 0 ]; then CMD=("$@"); else CMD=(bash gazebo/run_sim.sh); fi
exec $DOCKER run --rm $TTY --ipc=host $GPU_ARGS $X_ARGS $ENVS \
    -v "$ROOTDIR":/sim/nano_uav -w /sim/nano_uav $IMG "${CMD[@]}"
