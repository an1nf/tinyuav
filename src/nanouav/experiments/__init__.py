"""Experiment registry.

Every experiment is a module that can be executed with ``python -m <module> [args]`` (most of them
expose an argparse CLI; ``nanouav run <id> --help`` prints it). The registry gives each one a short,
stable id, a one-line summary and the resources it needs, so the CLI and the suite runner can list,
validate and schedule them.

Ids keep the code used in the research notes (``docs/research``): ``e*`` = original evaluation
plan (E0–E4), ``r*`` = research iterations R1–R15.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Experiment:
    id: str
    module: str
    summary: str
    needs: tuple = ('cpu',)            # 'cpu' | 'gpu' | 'tensorflow' | 'model:<variant>'
    tags: tuple = field(default=())
    legacy: str = ''                   # script name in the old flat layout


_E = [
    # ── evaluation plan E0–E4 ───────────────────────────────────────────────
    Experiment('e0c', 'nanouav.experiments.e0c_p3_latency', 'P3 policy: optical-flow trigger latency and detection GIF',
               tags=('policy',), legacy='e0c_p3_latency.py'),
    Experiment('e1a', 'nanouav.experiments.e1a_jitter_model', 'Physical jitter model of the inference loop (ESP32-S3)',
               tags=('hardware',), legacy='e1a_jitter_model.py'),
    Experiment('e1b', 'nanouav.experiments.e1b_energy', 'Energy budget and speed–safety–endurance trade-off',
               tags=('hardware',), legacy='e1b_energy.py'),
    Experiment('e2b', 'nanouav.experiments.e2b_tof_calibration', 'ToF-based scale calibration of the monocular CNN',
               tags=('perception',), legacy='e2b_tof_calibration.py'),
    Experiment('e2d', 'nanouav.experiments.e2d_early_exit', 'Early-exit proxy (optical flow + ToF) and latency gain',
               tags=('perception',), legacy='e2d_early_exit.py'),
    Experiment('e3', 'nanouav.experiments.e3_closed_loop', 'Closed loop: Gazebo-like world, disturbance kicks, Monte Carlo',
               tags=('sim', 'planning'), legacy='e3_closed_loop.py'),
    Experiment('e4', 'nanouav.experiments.e4_missions', 'Missions: waypoints, target following, pop-up obstacles',
               tags=('sim', 'missions'), legacy='e4_missions.py'),
    Experiment('figures', 'nanouav.experiments.figures', 'Overview figures (snapshot, optical flow, policies, benchmark)',
               tags=('figures',), legacy='run_all.py'),
    # ── research iterations ─────────────────────────────────────────────────
    Experiment('r01', 'nanouav.experiments.r01_lnca_footprint', 'LNCA vs U-Net: params, MACs, int8 arena, CPU latency',
               needs=('cpu', 'tensorflow'), tags=('perception', 'hardware'), legacy='r1_lnca_footprint.py'),
    Experiment('r02', 'nanouav.experiments.r02_follow_sensors', 'Target following with different sensor sets',
               tags=('sim', 'missions'), legacy='r2_follow_sensors.py'),
    Experiment('r03', 'nanouav.training.r3_lnca_train', 'Train/fine-tune M, M_ToF and LNCA variants (early ToF fusion)',
               needs=('gpu', 'tensorflow'), tags=('training',), legacy='r3_lnca_train.py'),
    Experiment('r04', 'nanouav.experiments.r04_speed_limit', 'Analytic speed limit vs perception range (Falanga/Loquercio)',
               tags=('analysis',), legacy='r4_speed_limit.py'),
    Experiment('r05', 'nanouav.experiments.r05_pole', 'Single-pole experiment: speed sweep, brake vs consistent law',
               tags=('sim', 'planning'), legacy='r5_pole.py'),
    Experiment('r06', 'nanouav.experiments.r06_corridor', 'Random corridors, brake vs consistent law (paper C2)',
               tags=('sim', 'planning'), legacy='r6_corridor_consistent.py'),
    Experiment('r07', 'nanouav.experiments.r07_camera_tilt', 'Camera tilt vs braking distance and ToF coverage',
               tags=('analysis',), legacy='r7_camera_tilt.py'),
    Experiment('r08', 'nanouav.experiments.r08_state_noise', 'Success vs state-estimation noise / bias (+drift fixes)',
               tags=('sim', 'estimation'), legacy='r8_state_noise.py'),
    Experiment('r10', 'nanouav.experiments.r10_dynamic', 'Dynamic obstacles (ball head-on / crossing), RF_DYN layer',
               tags=('sim', 'dynamic'), legacy='r10_dynamic.py'),
    Experiment('r11', 'nanouav.experiments.r11_vel_net', 'Learned velocity estimation from images (gen / train)',
               needs=('cpu', 'tensorflow'), tags=('estimation',), legacy='r11_vel_net.py'),
    Experiment('r11b', 'nanouav.experiments.r11b_flow_deck', 'Flow-deck style velocity from optical flow',
               tags=('estimation',), legacy='r11b_flow_deck.py'),
    Experiment('r12', 'nanouav.experiments.r12_scenarios', 'Scenario worlds shared with Gazebo; lane vs MH planner',
               tags=('sim', 'planning', 'gazebo'), legacy='g1_gazebo_world.py'),
    Experiment('r14', 'nanouav.experiments.r14_class_metrics', 'Class/region-wise depth metrics + safety rates (DDOS-style)',
               tags=('perception',), legacy='r14_class_metrics.py'),
    Experiment('r15', 'nanouav.experiments.r15_dyn_egomotion', 'False dynamic detections vs ego-motion (DroneMOT-style)',
               tags=('perception', 'dynamic'), legacy='r15_dyn_egomotion.py'),
    Experiment('r18', 'nanouav.agile.run', 'Loquercio 2021 recreation: SGM stereo, M-H expert, DAgger student (sgm|pilots|dagger|eval)',
               needs=('cpu', 'tensorflow'), tags=('planning', 'training', 'agile')),
    Experiment('r19', 'nanouav.follow.run', 'Follow a person: YOLO person detector (VisDrone), tracker + target lock, follow mission on R18 (prep-visdrone|train-det|eval-det|video|sim|dagger)',
               needs=('cpu', 'tensorflow'), tags=('perception', 'missions', 'agile')),
    Experiment('r25', 'nanouav.agile.diffphys', 'Recurrent depth policy trained through a differentiable point-mass simulator (Zhang et al. 2025) on the R18 forests (train; eval with r18 eval --pilots diffphys)',
               needs=('cpu', 'tensorflow'), tags=('training', 'agile')),
    Experiment('r22', 'nanouav.flightmare.run', 'The R18 student in Flightmare (Unity scenes of the paper) through a Python ZMQ bridge (probe|cloud|lines|eval|video|fake)',
               needs=('cpu', 'tensorflow'), tags=('agile', 'simulation')),
    Experiment('r20', 'nanouav.follow.run_balloon', 'Follow a colored balloon through a forest: color-blob detector, 3D target, follow mission on R18 (frames|sim|dagger|pick-color|video)',
               needs=('cpu', 'tensorflow'), tags=('perception', 'missions', 'agile')),
    # ── training & reporting ────────────────────────────────────────────────
    Experiment('train', 'nanouav.training.train_cnn', 'Train the XS/S/M/L depth CNNs and export int8 .tflite',
               needs=('gpu', 'tensorflow'), tags=('training',), legacy='train_cnn.py'),
    Experiment('train-tartanair', 'nanouav.training.tartanair', 'Training with TartanAir images mixed in',
               needs=('gpu', 'tensorflow'), tags=('training',), legacy='train.py'),
    Experiment('summarize', 'nanouav.results.summarize', 'Success rates with Wilson 95% CI for all result files',
               tags=('report',), legacy='summarize_results.py'),
    Experiment('summarize-gz', 'nanouav.results.summarize_gz', 'Gazebo episode table (+ same scenario offline)',
               tags=('report', 'gazebo'), legacy='gazebo/summarize_gz.py'),
]

REGISTRY: dict[str, Experiment] = {e.id: e for e in _E}
BY_LEGACY: dict[str, Experiment] = {e.legacy: e for e in _E if e.legacy}


def get(exp_id: str) -> Experiment:
    """Look up by id, legacy script name or module name."""
    if exp_id in REGISTRY:
        return REGISTRY[exp_id]
    if exp_id in BY_LEGACY or exp_id + '.py' in BY_LEGACY:
        return BY_LEGACY.get(exp_id) or BY_LEGACY[exp_id + '.py']
    for e in _E:
        if e.module == exp_id or e.module.rsplit('.', 1)[-1] == exp_id:
            return e
    raise KeyError(f'unknown experiment {exp_id!r}; see `nanouav list`')
