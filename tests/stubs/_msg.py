"""Plain attribute containers that mimic ROS message classes."""


class Msg:
    _fields = {}

    def __init__(self, **kw):
        for k, f in self._fields.items():
            setattr(self, k, f() if callable(f) else f)
        for k, v in kw.items():
            setattr(self, k, v)


class Vector3(Msg):
    _fields = {'x': 0.0, 'y': 0.0, 'z': 0.0}


class Quaternion(Msg):
    _fields = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
