"""Person detection and following on top of the Loquercio 2021 recreation (``nanouav.agile``), R19.

* ``people``    — walking people in the simulated worlds (body + head primitives, collision-free paths)
* ``detector``  — YOLO person detector (real images, ultralytics) and a simulated detector with misses,
                  jitter, occlusion and false positives (closed-loop tests in our renderer)
* ``tracker``   — multi-person tracker (ByteTrack-style two-stage association) + target lock with an
                  appearance model, so the drone keeps following *the same* person
* ``estimate``  — bounding box + stereo depth → 3D position; constant-velocity filter in the world
* ``mission``   — follow mission for the rollout: goal direction ω and speed for the student policy,
                  privileged reference for the expert / DAgger, metrics
* ``visdrone``  — VisDrone2019 → YOLO person dataset (aerial views)
"""
