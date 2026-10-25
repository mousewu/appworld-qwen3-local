"""Locate Slime's training entry point, without enforcing any framework version."""
import importlib.util
import os
from pathlib import Path


def find_training_script(environ=None, module_origin=None):
    environ = os.environ if environ is None else environ
    explicit = environ.get('SLIME_TRAIN_SCRIPT')
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f'SLIME_TRAIN_SCRIPT does not exist: {path}')
        return path
    root = environ.get('SLIME_DIR')
    if root:
        path = Path(root).expanduser().resolve() / 'train.py'
        if not path.is_file():
            raise ValueError(f'SLIME_DIR has no train.py: {path.parent}')
        return path
    if module_origin is None:
        spec = importlib.util.find_spec('slime')
        module_origin = spec.origin if spec else None
    if module_origin:
        package = Path(module_origin).resolve().parent
        for root in [package.parent, package]:
            candidate = root / 'train.py'
            if candidate.is_file():
                return candidate
    raise ValueError('Cannot locate Slime train.py. Set SLIME_DIR to your existing checkout '
                     'or SLIME_TRAIN_SCRIPT to its train.py. No version/commit is required.')


if __name__ == '__main__':
    try:
        print(find_training_script())
    except ValueError as exc:
        raise SystemExit(str(exc))
