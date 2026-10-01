"""Fake rclpy (tests only)."""
_ok = True


def init(*a, **k):
    global _ok
    _ok = True


def ok():
    return _ok


def shutdown(*a, **k):
    global _ok
    _ok = False


def spin_once(node, timeout_sec=None):
    pass


def spin(node):
    pass
