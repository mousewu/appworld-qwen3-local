"""Initialize Ray with project environment, then execute installed Slime unchanged."""
import os
import runpy
import signal
import sys

if __name__ == '__main__':
    def terminate(signum, frame):
        raise KeyboardInterrupt('Training termination requested')
    signal.signal(signal.SIGTERM, terminate)
    import ray
    script, *arguments = sys.argv[1:]
    keys = ['APPWORLD_GRPO_CONFIG', 'TENSORBOARD_DIR', 'PYTHONPATH',
            'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'CUDA_DEVICE_MAX_CONNECTIONS']
    ray.init(runtime_env={'env_vars': {k:os.environ[k] for k in keys if k in os.environ}})
    sys.argv = [script, *arguments]
    try:
        runpy.run_path(script, run_name='__main__')
    finally:
        ray.shutdown()
