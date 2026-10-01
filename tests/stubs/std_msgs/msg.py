from _msg import Msg


class Bool(Msg):
    _fields = {'data': False}


class Header(Msg):
    _fields = {'frame_id': '', 'stamp': None}
