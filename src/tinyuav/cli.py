"""tinyuav <group> <command> [options]; `tinyuav <group> -h` lists the commands."""
import importlib
import sys

from tinyuav import __version__

GROUPS = {
    'nav': ('tinyuav.nav.run', 'navigation in the simulated forests: baselines, student, evaluation, video'),
    'diffphys': ('tinyuav.nav.diffphys', 'navigation policy trained through differentiable physics'),
    'follow': ('tinyuav.follow.run', 'balloon following: policy, mission, video'),
    'flightmare': ('tinyuav.flightmare.run', 'the same in Flightmare (Unity)'),
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        for k, (_, h) in GROUPS.items():
            print(f'  {k:11s} {h}')
        return 0
    if argv[0] == '--version':
        print(__version__)
        return 0
    if argv[0] not in GROUPS:
        print(f'unknown group {argv[0]!r}; one of: {", ".join(GROUPS)}', file=sys.stderr)
        return 2
    mod = importlib.import_module(GROUPS[argv[0]][0])
    sys.argv = [f'tinyuav {argv[0]}'] + argv[1:]
    return mod.main(argv[1:]) or 0


if __name__ == '__main__':
    sys.exit(main())
