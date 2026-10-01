#!/bin/bash
# Launches the Gazebo container from the HOST (project root), with the GPU available for rendering.
#
#   bash gazebo/gz_docker.sh                         → simulation with windows (run_sim.sh)
#   WORLD=bosque SEED=2 bash gazebo/gz_docker.sh     → any run_sim.sh variable is passed to the container
#   bash gazebo/gz_docker.sh diag                    → GPU/EGL diagnostics inside the container
#   bash gazebo/gz_docker.sh bash                    → interactive shell
#   DOCKER="sudo docker" bash gazebo/gz_docker.sh    → if your user is not in the docker group
#   NO_GPU=1 ...                                     → software rendering (slow)
#
# Why: with only "--gpus all" the container gets the compute,utility (CUDA) capabilities but NOT
# graphics → no libEGL_nvidia/libGLX_nvidia → Mesa tries dri2 ("libEGL warning: failed to create
# dri2 screen") and falls back to llvmpipe: ~40–70 s startup and a low real-time factor.
# NVIDIA_DRIVER_CAPABILITIES=all mounts the graphics libraries of the host driver (nvidia-container-toolkit).
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
         T_MAX EXIT_ON_DONE HEADLESS RUN_TAG CONTROL_MODE YAW_SIGN MH_GRID_RES PLANNER_CFG \
         NANOUAV_MH_GRID_RES NANOUAV_FUSE2 NANOUAV_FAKE_CNN; do
    [ -n "${!v}" ] && ENVS="$ENVS -e $v=${!v}"
done
TTY=""; [ -t 0 ] && TTY="-it"

if [ "$1" = "diag" ]; then
    CMD='echo "== nvidia-smi"; nvidia-smi -L || echo "NO GPU: is nvidia-container-toolkit installed? --gpus all?";
echo "== EGL vendors (10_nvidia.json must be listed)"; ls /usr/share/glvnd/egl_vendor.d/ 2>/dev/null;
echo "== libEGL_nvidia"; ldconfig -p | grep -E "libEGL_nvidia|libGLX_nvidia" || echo "NO: graphics capability missing";
echo "== headless render test (10 s)"; source /opt/ros/jazzy/setup.bash;
timeout 25 gz sim -s -r --headless-rendering --iterations 2500 /sim/nano_uav/gazebo/worlds/nano_uav_world.sdf 2>&1 | grep -iE "egl|ogre|render|error" | head -8;
echo "(if there are no EGL errors, rendering uses the GPU)"'
    exec $DOCKER run --rm $TTY $GPU_ARGS -v "$ROOTDIR":/sim/nano_uav -w /sim/nano_uav $IMG bash -c "$CMD"
fi
if [ $# -gt 0 ]; then CMD=("$@"); else CMD=(bash gazebo/run_sim.sh); fi
exec $DOCKER run --rm $TTY --ipc=host $GPU_ARGS $X_ARGS $ENVS \
    -v "$ROOTDIR":/sim/nano_uav -w /sim/nano_uav $IMG "${CMD[@]}"
