"""AppWorld-only subprocess. Stdout is reserved for JSON-lines RPC."""
import contextlib
import importlib.metadata
import inspect
import json
import sys
import traceback


def world_kwargs(world_class, request):
    """Pass optional settings only when the installed SDK accepts them.

    Older releases lack some guard keywords. Omission keeps SDK defaults;
    it does not mean that an unsupported guard has been enabled.
    """
    parameters = inspect.signature(world_class).parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD
                         for p in parameters.values())
    required = dict(task_id=request['task_id'], experiment_name=request['experiment_name'])
    optional = dict(max_interactions=request['max_interactions'], random_seed=request['seed'],
                    raise_on_unsafe_syntax=True, raise_on_unsafe_execution=True)
    supported = {k: v for k, v in optional.items() if accepts_kwargs or k in parameters}
    skipped = sorted(set(optional) - set(supported))
    return {**required, **supported}, skipped


def main():
    world = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request['command']
            with contextlib.redirect_stdout(sys.stderr):
                from appworld import AppWorld, load_task_ids
                if command == 'list':
                    result = load_task_ids(request['split'])
                elif command == 'reset':
                    if world is not None:
                        world.close()
                    kwargs, skipped = world_kwargs(AppWorld, request)
                    if skipped:
                        print('AppWorld compatibility: unsupported optional arguments omitted '
                              '(installed SDK defaults retained): ' + ', '.join(skipped), file=sys.stderr)
                    world = AppWorld(**kwargs)
                    try:
                        version = importlib.metadata.version('appworld')
                    except importlib.metadata.PackageNotFoundError:
                        version = 'unknown'
                    result = dict(instruction=world.task.instruction,
                        supervisor=dict(world.task.supervisor), runtime=dict(
                            python=sys.executable, appworld_version=version,
                            appworld_class_file=inspect.getfile(AppWorld),
                            constructor_kwargs=kwargs, skipped_optional_kwargs=skipped))
                elif command == 'execute':
                    result = dict(observation=world.execute(request['code']),
                        declared_done=world.task_completed())
                elif command == 'evaluate':
                    tracker = world.evaluate()
                    result = dict(success=tracker.success,
                        passed_checks=tracker.pass_count, total_checks=tracker.num_tests)
                elif command == 'close':
                    if world is not None:
                        world.close()
                        world = None
                    result = None
                else:
                    raise ValueError(f'Unknown command: {command}')
            response = dict(ok=True, result=result)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = dict(ok=False, error=f'{type(exc).__name__}: {exc}')
        print(json.dumps(response, ensure_ascii=False), flush=True)
    if world is not None:
        with contextlib.redirect_stdout(sys.stderr):
            world.close()


if __name__ == '__main__':
    main()
