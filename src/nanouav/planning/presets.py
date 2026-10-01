"""Planner presets shared by the offline missions/experiments and the Gazebo node.

Keeping them in one place guarantees that ``nanouav run r12 --world json:…`` and Gazebo fly the same
planner with the same parameters (speed-dependent braking, reference limits, ``--cfg`` overrides).
This module only depends on the standard library, so the ROS node can import it cheaply.
"""


def fast_kw(v):
    """Planner keyword arguments and reference-limit profile as a function of the cruise speed."""
    fast = v > 0.8
    return ({'v_max': v, 'a_brake': 1.0 if fast else 0.8}, 'fast' if fast else 'normal')


def set_brake_law(planner):
    planner.speed_law = 'brake'
    planner.W_LANE = 0.6          # missions: returning to the segment line weighs more than the hysteresis
    planner.T_REACT = 0.6         # 0.1 s frame + 0.06 latency + jerk ramp + tracking delay
    planner.GAP_BIAS = 0.15       # map error (CNN + cell) observed at higher speed


def parse_cfg(text):
    """'INSPECT_TOF=1,M_ABS=0.02' → {'INSPECT_TOF': 1.0, 'M_ABS': 0.02} (planner attributes)."""
    return {k.strip(): float(v) for k, v in (x.split('=') for x in (text or '').split(',') if x.strip())}


def apply_cfg(planner, cfg):
    """Set planner attributes from a dict (or a 'K=v,…' string), keeping the type of each attribute.
    Unknown names raise AttributeError, so a typo does not silently run the default planner."""
    if isinstance(cfg, str):
        cfg = parse_cfg(cfg)
    for k, v in (cfg or {}).items():
        if not hasattr(planner, k):
            raise AttributeError(f'planner has no attribute {k!r} (--cfg / planner_cfg)')
        cur = getattr(planner, k)
        setattr(planner, k, type(cur)(v) if isinstance(cur, (bool, int, float)) else v)
    return planner
