#!/bin/bash
# R21 · the R18 student in Gazebo Harmonic (inside the container; from the host: bash gazebo/gz_docker.sh agile)
#
#   AGILE_MODEL=results/r18/student_train.tflite bash gazebo/run_agile.sh
#   AGILE_WORLD=forest AGILE_SEEDS=1000,1001,1002 AGILE_V=3 HEADLESS=true ...    → one episode per seed
#
# Variables: AGILE_MODEL (.tflite, required: `nanouav run r18 export --model ...` on the host)
#            AGILE_WORLD=forest|gap  AGILE_SEEDS=1000  AGILE_V=3
#            AGILE_TSCALE=chord|arc  AGILE_VCAP=0  AGILE_SELECT=snap|consistent  AGILE_GAIN=5
#            AGILE_KEEP_DEPTH=false (true: also save the network input, for `r18 video --from-gz`)
#            HEADLESS=true (no GUI)  RUN_TAG=...
# Results: results/gz_runs/agile/<model>_<world>_s<seed>_v<v>[_tag].json (+ a summary at the end)
set -e
source /opt/ros/jazzy/setup.bash
ROOT=${NANO_UAV_ROOT:-/sim/nano_uav}
export NANO_UAV_ROOT=$ROOT NANOUAV_HOME=$ROOT
export GZ_SIM_RESOURCE_PATH=$ROOT/gazebo/models:$GZ_SIM_RESOURCE_PATH
export PYTHONPATH=$ROOT/src:$PYTHONPATH
export GZ_PARTITION=agile_$(hostname)_$$
export ROS_AUTOMATIC_DISCOVERY_RANGE=${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}
W=${AGILE_WORLD:-forest}
V=${AGILE_V:-3}
MODEL=${AGILE_MODEL:-}
[[ "$MODEL" = /* ]] || MODEL=$ROOT/$MODEL
if [ ! -f "$MODEL" ]; then
    echo "AGILE_MODEL not found: $MODEL"
    echo "export it on the host first: nanouav run r18 export --model results/r18/student_train.keras"
    exit 1
fi

give_back () {
    [ -n "${HOST_UID:-}" ] && [ "$(id -u)" = "0" ] || return 0
    for d in $ROOT/gazebo/worlds/gen $ROOT/gazebo/models/nano_textures $ROOT/gazebo/models/agile_x3 $ROOT/results/gz_runs; do
        [ -e "$d" ] && { chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" "$d" 2>/dev/null || true; }
    done
    find $ROOT/src $ROOT/gazebo -name __pycache__ -user 0 -exec chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" {} + 2>/dev/null || true
    return 0
}

run_one () {   # seed
    local S=$1 TAG=agile_${W}_s$1
    local SDF=$ROOT/gazebo/worlds/gen/$TAG.sdf
    if [ ! -f "$SDF" ] || [ ! -f "$ROOT/gazebo/models/agile_x3/model.sdf" ] || [ -n "$REGEN" ]; then
        python3 $ROOT/gazebo/tools/gen_world.py --agile $W --seed $S
        give_back
    fi
    echo "=== R18 student · Gazebo · $W seed $S · $V m/s · $(basename $MODEL) ==="
    if [ -n "$DISPLAY" ] && [ "${HEADLESS:-false}" != "true" ]; then
        gz sim $SDF -r &
    else
        gz sim $SDF -r -s --headless-rendering &
    fi
    GZ_PID=$!
    for i in $(seq 1 120); do
        gz service -l 2>/dev/null | grep -q "/world/nano_uav_world/create" && break
        sleep 1
    done
    echo "world loaded in ~${i} s"
    ros2 run ros_gz_bridge parameter_bridge \
        /nano_uav/gazebo/command/twist@geometry_msgs/msg/Twist@gz.msgs.Twist \
        /nano_uav/enable@std_msgs/msg/Bool@gz.msgs.Boolean \
        /model/nano_uav/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry \
        /agile/left/image@sensor_msgs/msg/Image@gz.msgs.Image \
        /agile/right/image@sensor_msgs/msg/Image@gz.msgs.Image \
        "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock" > /tmp/bridge_agile.log 2>&1 &
    BR_PID=$!
    sleep 2
    OPT=()
    [ -n "${RUN_TAG:-}" ] && OPT+=(-p "run_tag:=$(basename $MODEL .tflite)_${W}_s${S}_v${V}_${RUN_TAG}")
    python3 $ROOT/gazebo/nodes/agile_node.py --ros-args -p use_sim_time:=true -p world:=$W -p seed:=$S \
        -p v_des:=$V -p model:=$MODEL -p tscale:=${AGILE_TSCALE:-chord} -p vcap:=${AGILE_VCAP:-0.0} \
        -p select:=${AGILE_SELECT:-snap} -p gain:=${AGILE_GAIN:-5.0} -p keep_depth:=${AGILE_KEEP_DEPTH:-false} \
        -p exit_on_done:=true "${OPT[@]}" &
    NODE_PID=$!
    wait $NODE_PID || true
    kill $BR_PID $GZ_PID 2>/dev/null; sleep 1; kill -9 $GZ_PID 2>/dev/null; pkill -f "gz sim" 2>/dev/null || true
    sleep 1
}

trap 'kill $NODE_PID $BR_PID $GZ_PID 2>/dev/null; pkill -f "gz sim" 2>/dev/null; give_back; exit 0' SIGINT SIGTERM
for S in $(echo ${AGILE_SEEDS:-1000} | tr ',' ' '); do
    run_one $S
done
give_back
python3 -c "
import glob, json, os, sys
rows = [json.load(open(f))['result'] for f in sorted(glob.glob(os.path.join(sys.argv[1], 'results/gz_runs/agile/*.json')))]
rows = [r for r in rows if r.get('world') == sys.argv[2] and abs(r.get('v_des', 0) - float(sys.argv[3])) < 1e-6]
print(f'{len(rows)} Gazebo episodes ({sys.argv[2]}, {sys.argv[3]} m/s): '
      f'{sum(r[\"success\"] for r in rows)} success, {sum(r[\"crashed\"] for r in rows)} crash')
for r in rows:
    print(f\"  s{r['seed']} {r['model']} {r['status']:8s} {r['time_s']:6.1f} s  clear {r['min_clearance_m']:.2f} m  v {r['mean_speed']:.1f}/{r['max_speed']:.1f} m/s  {r['opts']}\")
" $ROOT $W $V
