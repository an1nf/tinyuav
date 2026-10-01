#!/bin/bash
# Corridas por lotes en Gazebo (sin ventanas), una por contenedor; resultados en assets/gz_runs/*.json
#
#   bash gazebo/batch_eval.sh                                   → obstaculos,bosque,rendijas,interior × semillas 0–4 × mh
#   WORLDS=bosque,rendijas SEEDS=0,1,2 PLANNERS=mh,lane bash gazebo/batch_eval.sh
#   PAR=2 ...                                                   → contenedores en paralelo (cada uno usa ~1–2 núcleos + GPU)
#   python3 gazebo/summarize_gz.py                              → tabla con IC95 (Wilson) + comparación con el simulador propio
WORLDS=${WORLDS:-obstaculos,bosque,rendijas,interior}
SEEDS=${SEEDS:-0,1,2,3,4}
PLANNERS=${PLANNERS:-mh}
PAR=${PAR:-1}
TAG=${RUN_TAG:-batch_$(date +%m%d_%H%M)}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HERE/../logs/gz"
jobs_file=$(mktemp)
for w in ${WORLDS//,/ }; do for s in ${SEEDS//,/ }; do for p in ${PLANNERS//,/ }; do
    echo "$w $s $p" >> "$jobs_file"
done; done; done
echo "$(wc -l < "$jobs_file") episodios · paralelo=$PAR · etiqueta=$TAG"
run_one () {
    read w s p <<< "$1"
    log="$HERE/../logs/gz/${w}_s${s}_${p}.log"
    WORLD=$w SEED=$s PLANNER=$p HEADLESS=true EXIT_ON_DONE=true RUN_TAG=$TAG \
        timeout ${EP_TIMEOUT:-400} bash "$HERE/gz_docker.sh" > "$log" 2>&1
    echo "[$w s$s $p] $(grep -o 'EPISODIO: [a-z_]* · t=[0-9.]* s' "$log" | tail -1)"
}
export -f run_one; export HERE TAG
xargs -d '\n' -P "$PAR" -I{} bash -c 'run_one "$@"' _ {} < "$jobs_file"
rm -f "$jobs_file"
PYTHONPATH="$HERE/../src" python3 -m nanouav.results.summarize_gz --tag "$TAG"
