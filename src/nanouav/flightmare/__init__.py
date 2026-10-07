"""R22 · Flightmare (the simulator of Loquercio et al. 2021) as renderer for nanouav.agile.

Flightmare's Unity standalone renders; flightlib's physics is not used: our quadrotor model, controller,
expert and student stay the same, and only the images come from Unity. `bridge` speaks Flightmare's ZMQ
protocol from Python (no flightlib build, no ROS), `cloud` turns the Unity scene point cloud into the
privileged map (collisions, global path, expert), `run` is `nanouav run r22`.
"""
