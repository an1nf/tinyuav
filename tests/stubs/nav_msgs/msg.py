from _msg import Msg
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance


class Odometry(Msg):
    _fields = {'pose': PoseWithCovariance, 'twist': TwistWithCovariance}
