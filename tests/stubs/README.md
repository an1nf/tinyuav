Minimal fakes of the ROS 2 Python API (rclpy + message types) used by `gazebo/nodes/*.py`.
They let the tests import and drive the nodes without a ROS installation: publishers record
what they publish, timers/subscriptions are kept so the test can call the callbacks by hand.
Only the attributes the nodes actually use are implemented.
