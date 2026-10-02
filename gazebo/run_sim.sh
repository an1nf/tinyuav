#!/bin/bash
# Nano-UAV · Gazebo Harmonic + ROS 2 Jazzy (inside the container)
#
#   bash gazebo/run_sim.sh                              → generated world 'obstaculos' seed 0, MH planner
#   WORLD=bosque SEED=3 bash gazebo/run_sim.sh          → scenarios: obstaculos bosque rendijas interior corredor dinamico
#   WORLD=bosque WORLD_KW="gap=1.0,density=0.5" ...     → scenario parameters (see nanouav/sim/scenarios.py)
#   WORLD=base bash gazebo/run_sim.sh                   → original world (gazebo/worlds/nano_uav_world.sdf)
#   HEADLESS=true EXIT_ON_DONE=true ...                 → no windows; exits when the episode ends (batches)
#
# Other variables: VARIANT=M|M_ToF  PROFILE=x3|nano  PLANNER=mh|lane  CONSISTENT=true|false  RF_DYN=true|false
#                  V_CRUISE=0.5  GOAL_X/GOAL_Y (override B)  LOOP=true  T_MAX=120  RUN_TAG=...  CONTROL_MODE=motors
#                  PLANNER_CFG="K=v,..." (planner attributes, same as `nanouav run r12 --cfg`)
#                  A world reset from the Gazebo GUI starts a new measured episode (results/gz_runs/*_epN.json)
set -e
source /opt/ros/jazzy/setup.bash
ROOT=${NANO_UAV_ROOT:-/sim/nano_uav}
export NANO_UAV_ROOT=$ROOT
export GZ_SIM_RESOURCE_PATH=$ROOT/gazebo/models:$GZ_SIM_RESOURCE_PATH
# Isolate every container: with a shared partition / ROS domain, two simulations running at the same time
# (batch_eval.sh PAR>1, or a leftover container) see each other's /clock and topics → negative real-time
# factor, timeouts that never fire, commands from the other drone.
export GZ_PARTITION=nano_uav_$(hostname)_$$
export ROS_AUTOMATIC_DISCOVERY_RANGE=${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}
export PYTHONPATH=$ROOT/src:$PYTHONPATH
export NANOUAV_HOME=$ROOT
MODE=${CONTROL_MODE:-velocity}
WORLD=${WORLD:-obstaculos}
SEED=${SEED:-0}

# The container runs as root: without this, the generated worlds, textures and results belong to root
# and `nanouav gazebo gen-world` on the host fails with "Permission denied". gz_docker.sh passes HOST_UID.
give_back () {
    [ -n "${HOST_UID:-}" ] && [ "$(id -u)" = "0" ] || return 0
    for d in $ROOT/gazebo/worlds/gen $ROOT/gazebo/models/nano_textures $ROOT/results/gz_runs; do
        [ -e "$d" ] && { chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" "$d" 2>/dev/null || true; }
    done
    find $ROOT/src $ROOT/gazebo -name __pycache__ -user 0 -exec chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" {} + 2>/dev/null || true
    return 0
}

# ── world ──────────────────────────────────────────────────────────────────
WJSON=""
if [ "$WORLD" = "base" ]; then
    SDF=$ROOT/gazebo/worlds/nano_uav_world.sdf
    DEF_GX=4.5; DEF_GY=0.0
else
    KWARGS=""; KTAG=""
    if [ -n "$WORLD_KW" ]; then
        KWARGS="--kw $WORLD_KW"
        KTAG=$(python3 -c "import sys; print(''.join('_%s%g' % (k, float(v)) for k, v in (x.split('=') for x in sys.argv[1].split(',') if x)))" "$WORLD_KW")
    fi
    TAG=${WORLD}_s${SEED}${KTAG}
    SDF=$ROOT/gazebo/worlds/gen/$TAG.sdf
    WJSON=$ROOT/gazebo/worlds/gen/$TAG.json
    if [ ! -f "$SDF" ] || [ ! -f "$WJSON" ] || [ -n "$REGEN" ]; then
        echo "generating world $TAG ..."
        python3 $ROOT/gazebo/tools/gen_world.py --scenario $WORLD --seed $SEED $KWARGS
        give_back
    fi
    read DEF_GX DEF_GY DYN <<< $(python3 -c "import json,sys; s=json.load(open(sys.argv[1])); print(s['B'][0], s['B'][1], len(s['dynamic']))" "$WJSON")
fi
GOAL_X=${GOAL_X:-$DEF_GX}; GOAL_Y=${GOAL_Y:-$DEF_GY}

echo "=== Nano-UAV · Gazebo · world=$WORLD seed=$SEED · planner=${PLANNER:-mh} · control=$MODE ==="
if [ -n "$DISPLAY" ] && [ "${HEADLESS:-false}" != "true" ]; then
    gz sim $SDF -r &
else
    gz sim $SDF -r -s --headless-rendering &
fi
GZ_PID=$!

# wait until the world is loaded (with software rendering this can take > 30 s)
for i in $(seq 1 120); do
    gz service -l 2>/dev/null | grep -q "/world/nano_uav_world/create" && break
    sleep 1
done
echo "world loaded in ~${i} s"

BRIDGES="/nano_uav/gazebo/command/twist@geometry_msgs/msg/Twist@gz.msgs.Twist
/nano_uav/gazebo/command/motor_speed@actuator_msgs/msg/Actuators@gz.msgs.Actuators
/nano_uav/enable@std_msgs/msg/Bool@gz.msgs.Boolean
/model/nano_uav/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry
/nano_uav/imu@sensor_msgs/msg/Imu@gz.msgs.IMU
/nano_uav/camera/image@sensor_msgs/msg/Image@gz.msgs.Image
/nano_uav/depth/image@sensor_msgs/msg/Image@gz.msgs.Image
/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"
if [ -n "$WJSON" ] && [ "${DYN:-0}" -gt 0 ]; then
    for n in $(python3 -c "import json,sys; print(' '.join(d['name'] for d in json.load(open(sys.argv[1]))['dynamic']))" "$WJSON"); do
        BRIDGES="$BRIDGES
/model/$n/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist
/model/$n/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry"
    done
fi
ros2 run ros_gz_bridge parameter_bridge $BRIDGES > /tmp/bridge.log 2>&1 &
BRIDGE_PID=$!
sleep 2

# B marker only if the world's goal is overridden
spawn_marker () {   # name x y r g b
    MSDF="<sdf version='1.9'><model name='$1'><static>true</static><pose>$2 $3 0.008 0 0 0</pose><link name='l'>\
<visual name='v'><geometry><cylinder><radius>0.25</radius><length>0.004</length></cylinder></geometry>\
<material><ambient>$4 $5 $6 1</ambient><diffuse>$4 $5 $6 1</diffuse><emissive>$4 $5 $6 1</emissive></material></visual>\
</link></model></sdf>"
    gz service -s /world/nano_uav_world/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean \
        --timeout 5000 --req "sdf: \"$MSDF\" allow_renaming: true" 2>/dev/null | grep -q true \
        && echo "marker $1 at ($2, $3)" || echo "warning: could not create marker $1"
}
if [ "$GOAL_X" != "$DEF_GX" ] || [ "$GOAL_Y" != "$DEF_GY" ]; then
    spawn_marker meta_B_custom $GOAL_X $GOAL_Y 0.1 0.9 0.1
fi

DYN_PID=""
if [ -n "$WJSON" ] && [ "${DYN:-0}" -gt 0 ]; then
    python3 $ROOT/gazebo/nodes/dyn_obstacles_node.py --ros-args -p world_json:=$WJSON &
    DYN_PID=$!
fi
FC_PID=""
if [ "$MODE" = "motors" ]; then
    python3 $ROOT/gazebo/nodes/flight_controller_node.py --ros-args -p yaw_sign:=${YAW_SIGN:-1} &
    FC_PID=$!
fi
# ROS 2 (Jazzy) rejects empty overrides such as `-p run_tag:=` → only pass string params that are set
OPT_ARGS=()
[ -n "$WJSON" ] && OPT_ARGS+=(-p "world_json:=$WJSON")
[ -n "${RUN_TAG:-}" ] && OPT_ARGS+=(-p "run_tag:=$RUN_TAG")
[ -n "${PLANNER_CFG:-}" ] && OPT_ARGS+=(-p "planner_cfg:=$PLANNER_CFG")     # e.g. "MAP_DECAY_FOV=1,MAP_HIT_WINS=1"
python3 $ROOT/gazebo/nodes/perception_node.py --ros-args -p control_mode:=$MODE \
    -p variant:=${VARIANT:-M} -p profile:=${PROFILE:-x3} -p consistent:=${CONSISTENT:-false} \
    -p rf_dyn:=${RF_DYN:-false} -p v_cruise:=${V_CRUISE:-0.5} \
    -p planner:=${PLANNER:-mh} -p goal:=${GOAL:-true} -p goal_x:=$GOAL_X -p goal_y:=$GOAL_Y -p loop:=${LOOP:-false} \
    -p t_max:=${T_MAX:-0.0} -p exit_on_done:=${EXIT_ON_DONE:-false} \
    -p hud:=$([ "${HEADLESS:-false}" = "true" ] && echo false || echo true) "${OPT_ARGS[@]}" &
NODE_PID=$!

cleanup () { kill $NODE_PID $DYN_PID $FC_PID $BRIDGE_PID $GZ_PID 2>/dev/null; sleep 1; kill -9 $GZ_PID 2>/dev/null; pkill -f "gz sim" 2>/dev/null; give_back; true; }
trap "cleanup; exit 0" SIGINT SIGTERM
echo "Ctrl+C to stop"
if [ "${EXIT_ON_DONE:-false}" = "true" ]; then
    wait $NODE_PID || true
    cleanup
else
    wait $GZ_PID || true
    cleanup
fi
