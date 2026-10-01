#!/bin/bash
# Batch runs in Gazebo (no windows), one per container; results in results/gz_runs/*.json
#
#   bash gazebo/batch_eval.sh                                   → obstaculos,bosque,rendijas,interior × seeds 0–4 × mh
#   WORLDS=bosque,rendijas SEEDS=0,1,2 PLANNERS=mh,lane bash gazebo/batch_eval.sh
#   PAR=2 ...                                                   → containers in parallel (each uses ~1–2 cores + GPU)
#   nanouav summarize --gazebo                                  → table with 95% CI (Wilson) + comparison with our own simulator
WORLDS=${WORLDS:-obstaculos,bosque,rendijas,interior}
SEEDS=${SEEDS:-0,1,2,3,4}
PLANNERS=${PLANNERS:-mh}
PAR=${PAR:-1}
TAG=${RUN_TAG:-batch_$(date +%m%d_%H%M)}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HERE/../results/gz_logs"
jobs_file=$(mktemp)
for w in ${WORLDS//,/ }; do for s in ${SEEDS//,/ }; do for p in ${PLANNERS//,/ }; do
    echo "$w $s $p" >> "$jobs_file"
done; done; done
echo "$(wc -l < "$jobs_file") episodes · parallel=$PAR · tag=$TAG"
run_one () {
    read w s p <<< "$1"
    log="$HERE/../results/gz_logs/${w}_s${s}_${p}.log"
    WORLD=$w SEED=$s PLANNER=$p HEADLESS=true EXIT_ON_DONE=true RUN_TAG=$TAG \
        timeout ${EP_TIMEOUT:-400} bash "$HERE/gz_docker.sh" > "$log" 2>&1
    echo "[$w s$s $p] $(grep -o 'EPISODIO: [a-z_]* · t=[0-9.]* s' "$log" | tail -1)"
}
export -f run_one; export HERE TAG
xargs -d '\n' -P "$PAR" -I{} bash -c 'run_one "$@"' _ {} < "$jobs_file"
rm -f "$jobs_file"
PYTHONPATH="$HERE/../src" python3 -m nanouav.results.summarize_gz --tag "$TAG"
