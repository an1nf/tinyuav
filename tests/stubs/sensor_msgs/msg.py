from _msg import Msg, Vector3, Quaternion


class Image(Msg):
    _fields = {'height': 0, 'width': 0, 'encoding': 'mono8', 'data': b''}


class Imu(Msg):
    _fields = {'orientation': Quaternion, 'angular_velocity': Vector3, 'linear_acceleration': Vector3}
