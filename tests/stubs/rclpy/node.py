"""Fake rclpy.node.Node: records publishers, subscriptions and timers."""
import time


class _Param:
    def __init__(self, value):
        self.value = value


class _Logger:
    def __init__(self):
        self.lines = []

    def _log(self, lvl, msg):
        self.lines.append((lvl, str(msg)))

    def info(self, m):
        self._log('info', m)

    def warn(self, m):
        self._log('warn', m)

    warning = warn

    def error(self, m):
        self._log('error', m)

    def debug(self, m):
        self._log('debug', m)


class _Pub:
    def __init__(self, typ, topic):
        self.typ, self.topic, self.msgs = typ, topic, []

    def publish(self, m):
        self.msgs.append(m)


class _Time:
    def __init__(self, ns):
        self.nanoseconds = ns

    def to_msg(self):
        return (self.nanoseconds // 10**9, self.nanoseconds % 10**9)


class _Clock:
    def __init__(self, node):
        self.node = node

    def now(self):
        return _Time(int(self.node._fake_time_ns if self.node._fake_time_ns is not None else time.monotonic_ns()))


class Node:
    #: parameter overrides applied by declare_parameter (set by the test before instantiating)
    OVERRIDES = {}

    def __init__(self, name, *a, **k):
        self._name, self._params, self._logger = name, {}, _Logger()
        self.publishers, self.subscriptions, self.timers = {}, {}, []
        self._fake_time_ns = None

    def declare_parameter(self, name, default=None):
        self._params[name] = _Param(Node.OVERRIDES.get(name, default))
        return self._params[name]

    def get_parameter(self, name):
        return self._params[name]

    def create_publisher(self, typ, topic, qos):
        p = _Pub(typ, topic)
        self.publishers[topic] = p
        return p

    def create_subscription(self, typ, topic, cb, qos):
        self.subscriptions[topic] = cb
        return cb

    def create_timer(self, period, cb):
        self.timers.append((period, cb))
        return cb

    def get_logger(self):
        return self._logger

    def get_clock(self):
        return _Clock(self)

    def destroy_node(self):
        pass
