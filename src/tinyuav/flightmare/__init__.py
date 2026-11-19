"""Flightmare's Unity player as the renderer of the closed loop; dynamics, control and pilots stay ours.

bridge  ZMQ client of the player (no flightlib, no ROS)
arena   scanned static scene plus spawned crates with exact geometry
scan    map of a scene from Unity's depth images
follow  balloons painted into Unity's stereo images
fake    our renderer behind the same protocol (tests)
"""
