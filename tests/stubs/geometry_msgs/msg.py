from _msg import Msg, Vector3, Quaternion  # noqa: F401


class Point(Msg):
    _fields = {'x': 0.0, 'y': 0.0, 'z': 0.0}


class Twist(Msg):
    _fields = {'linear': Vector3, 'angular': Vector3}


class Pose(Msg):
    _fields = {'position': Point, 'orientation': Quaternion}


class PoseWithCovariance(Msg):
    _fields = {'pose': Pose}


class TwistWithCovariance(Msg):
    _fields = {'twist': Twist}
