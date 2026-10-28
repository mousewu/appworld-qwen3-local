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


def find_async_training_script(implementation='auto', environ=None, module_origin=None):
    env = os.environ if environ is None else environ
    script = find_training_script(env, module_origin)
    legacy = script.parent / 'train_async.py'
    if implementation in ('auto', 'pipeline') and legacy.is_file():
        return legacy, 'pipeline'
    if implementation == 'pipeline':
        raise ValueError('Installed Slime has no train_async.py; use --async-implementation fully '
                         'only if its native fully_async_rollout API is available')
    if not (script.parent/'slime/rollout/fully_async_rollout.py').is_file():
        import importlib.util
        spec = importlib.util.find_spec('slime')
        package = Path(spec.origin).parent if spec and spec.origin else Path('/nonexistent')
        if not (package/'rollout/fully_async_rollout.py').is_file():
            raise ValueError('No native Slime async entry/rollout implementation found')
    return script.parent/'train.py', 'fully'


if __name__ == '__main__':
    try:
        print(find_training_script())
    except ValueError as exc:
        raise SystemExit(str(exc))
