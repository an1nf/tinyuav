from _msg import Msg
from std_msgs.msg import Header


class Actuators(Msg):
    _fields = {'header': Header, 'velocity': list, 'position': list, 'normalized': list}
