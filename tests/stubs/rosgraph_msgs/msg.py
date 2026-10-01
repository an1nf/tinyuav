from _msg import Msg


class Time(Msg):
    _fields = {'sec': 0, 'nanosec': 0}


class Clock(Msg):
    _fields = {'clock': Time}
