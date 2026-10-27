"""Local Qwen3 ReAct runner based on AppWorld's minimal_agent.ipynb."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import ipaddress
import json
import os
from numbers import Integral
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import uuid

from inference_batching import BatchingBackend

HERE = Path(__file__).resolve().parent


class ContextLimit(RuntimeError):
    pass


def normalize_input_ids(encoded):
    """Extract one integer token sequence from HF or native Tokenizers output."""
    if isinstance(encoded, Mapping):
        if 'input_ids' not in encoded:
            raise TypeError('Tokenizer mapping has no input_ids')
        encoded = encoded['input_ids']
    if hasattr(encoded, 'ids'):
        encoded = encoded.ids
    elif hasattr(encoded, 'tolist'):
        encoded = encoded.tolist()
    # A tokenizer may return a single-element batch containing an Encoding.
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1:
        first = encoded[0]
        if hasattr(first, 'ids') or isinstance(first, (list, tuple)) or hasattr(first, 'tolist'):
            return normalize_input_ids(first)
    if not isinstance(encoded, (list, tuple)) or not encoded:
        raise TypeError(f'Expected a nonempty token ID sequence, got {type(encoded).__name__}')
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in encoded):
        raise TypeError('Tokenizer must return exactly one sequence of nonnegative integer token IDs')
    return [int(x) for x in encoded]


def chat_input_ids(tokenizer, messages):
    # Render then encode explicitly: apply_chat_template(tokenize=True) has
    # returned different container types across Transformers versions.
    text = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    if not isinstance(text, str):
        raise TypeError(f'Expected rendered chat text, got {type(text).__name__}')
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    return normalize_input_ids(encoded)


def model_dtype_kwargs(version, dtype):
    match = re.match(r'(\d+)\.(\d+)', version)
    modern = match and tuple(map(int, match.groups())) >= (4, 56)
    return {'dtype' if modern else 'torch_dtype': dtype}


def prompt_messages(supervisor, instruction):
    # Split the trusted template BEFORE inserting task text, so task content cannot
    # introduce new message roles by containing 'ASSISTANT:' or 'USER:'.
    from jinja2 import Environment, StrictUndefined
    template = (HERE / 'prompt_template.txt').read_text()
    chunks = re.split(r'(?m)^(USER|ASSISTANT|SYSTEM):\n', template)
    if chunks[0].strip():
        raise ValueError('Prompt template has an unassigned prefix')
    env = Environment(undefined=StrictUndefined, autoescape=False)
    return [dict(role=chunks[i].lower(), content=env.from_string(chunks[i+1]).render(
        supervisor=supervisor, instruction=instruction).strip())
        for i in range(1, len(chunks), 2)]


def extract_code(text):
    """Accept raw Python or exactly one fenced/code-tag block; never repair code."""
    text = text.strip()
    # Native SGLang may retain decoded EOS markers (no_stop_trim/skip_special_tokens).
    # Remove only trailing Qwen chat terminators for execution. Sampled training
    # tokens/logprobs are owned by the rollout adapter and must remain untouched.
    while text.endswith(('<|im_end|>', '<|endoftext|>')):
        marker = '<|im_end|>' if text.endswith('<|im_end|>') else '<|endoftext|>'
        text = text[:-len(marker)].rstrip()
    if not text:
        raise ValueError('Empty model response')
    if '<think>' in text or '</think>' in text:
        raise ValueError('Unexpected thinking output; disable thinking in the server')
    fenced = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S)
    tagged = re.findall(r'<code>(.*?)</code>', text, re.S)
    if fenced or tagged:
        if len(fenced) + len(tagged) != 1:
            raise ValueError('Expected exactly one code block')
        text = (fenced or tagged)[0].strip()
    ast.parse(text)
    return text


def format_error_feedback(exc):
    location = ''
    if isinstance(exc, SyntaxError):
        location = f'\nPython syntax error at line {exc.lineno}: {exc.msg}'
        if exc.text:
            location += '\nProblematic line: ' + exc.text.strip()
    else:
        location = '\nFormat error: ' + str(exc)
    return ('[AGENT OUTPUT FORMAT ERROR — CODE NOT EXECUTED]\n'
        'Your previous response was rejected before execution. NO Python statement or API call '
        'from that response ran. This is NOT an AppWorld API error, authentication failure, '
        'or evidence that an API/documentation is unavailable.' + location + '\n'
        'Retry the intended operation with valid Python only. Put EVERY explanation line behind # '
        'or omit explanations. Do not change your API plan merely because of this formatting error. '
        'Return one complete Python code block; do not invent the result of the unexecuted call.')


def execution_feedback(observation):
    failed = observation.lstrip().startswith('Execution failed')
    header = ('[APPWORLD EXECUTION ERROR — CODE WAS SUBMITTED]' if failed else
              '[APPWORLD EXECUTION OUTPUT — CODE WAS SUBMITTED]')
    return header + '\n' + observation, failed


def build_summary(results, planned):
    n = len(results)
    successes = [r for r in results if r['success']]
    steps = sum(r['steps'] for r in results)
    executed = sum(r.get('executed_steps', 0) for r in results)
    return dict(results=results, planned_tasks=planned, tasks=n, unique_tasks=len({r['task_id'] for r in results}),
        evaluated=sum(r['evaluated'] for r in results), successes=len(successes),
        failures=n-len(successes), errors=sum(r['termination'] == 'error' for r in results),
        success_rate=len(successes)/n if n else 0.,
        # Errors remain in the denominator; no silent exclusion of failed runs.
        termination_counts=dict(Counter(r['termination'] for r in results)),
        failure_termination_counts=dict(Counter(r['termination'] for r in results if not r['success'])),
        total_turns=steps, average_turns=steps/n if n else 0.,
        average_success_turns=(sum(r['steps'] for r in successes)/len(successes) if successes else None),
        total_executed_steps=executed,
        format_errors=sum(r.get('format_errors', 0) for r in results),
        format_error_rate=(sum(r.get('format_errors', 0) for r in results)/steps if steps else 0.),
        execution_errors=sum(r.get('execution_errors', 0) for r in results),
        execution_error_rate=(sum(r.get('execution_errors', 0) for r in results)/executed if executed else 0.),
        authentication_errors=sum(r.get('authentication_errors', 0) for r in results),
        seconds=sum(r['seconds'] for r in results),
        completed=n == planned)


def save_results(out, results, planned, wall_seconds=None, inference=None):
    summary = build_summary(results, planned)
    if wall_seconds is not None:
        summary['wall_seconds'] = wall_seconds
        summary['episodes_per_minute'] = len(results) * 60 / wall_seconds if wall_seconds else 0.
        summary['completion_tokens_per_second'] = (
            sum(r.get('completion_tokens', 0) for r in results)/wall_seconds if wall_seconds else 0.)
    if inference is not None:
        summary['inference'] = inference
    temp = out / 'summary.json.tmp'
    temp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    temp.replace(out / 'summary.json')
    fields = ['task_id', 'success', 'evaluated', 'declared_done', 'termination', 'steps',
              'executed_steps', 'format_errors', 'execution_errors', 'authentication_errors',
              'completion_tokens', 'model_wait_seconds', 'environment_seconds',
              'passed_checks', 'total_checks', 'seconds', 'error']
    with (out / 'results.csv.tmp').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    (out / 'results.csv.tmp').replace(out / 'results.csv')
    return summary


class WorldProcess:
    """JSON-lines RPC; each child owns one world and no model weights."""
    def __init__(self, python, root, timeout):
        self.timeout = timeout
        self.responses = queue.Queue()
        env = os.environ.copy()
        env['APPWORLD_ROOT'] = str(root)
        self.proc = subprocess.Popen([python, '-u', str(HERE / 'world_worker.py')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
            cwd=root, env=env)
        def reader():
            for line in self.proc.stdout:
                try:
                    self.responses.put(json.loads(line))
                except json.JSONDecodeError:
                    self.responses.put({'ok': False, 'error': 'Invalid worker protocol output'})
            self.responses.put({'ok': False, 'error': 'AppWorld worker exited; inspect stderr'})
        self.reader_thread = threading.Thread(target=reader, daemon=True)
        self.reader_thread.start()

    def call(self, command, **kwargs):
        try:
            self.proc.stdin.write(json.dumps(dict(command=command, **kwargs)) + '\n')
            self.proc.stdin.flush()
            response = self.responses.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError(f'AppWorld worker exceeded {self.timeout}s during {command}')
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError('AppWorld worker unavailable') from exc
        if not response.get('ok'):
            raise RuntimeError(response.get('error', 'Unknown AppWorld error'))
        return response.get('result')

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.reader_thread.join(timeout=2)
        for stream in (self.proc.stdin, self.proc.stdout):
            if stream:
                stream.close()


class TransformersBackend:
    def __init__(self, args):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = Path(args.model).expanduser().resolve()
        if not model.is_dir():
            raise ValueError('--model must be a downloaded local model directory')
        self.args, self.torch = args, torch
        device = args.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else (
                'mps' if torch.backends.mps.is_available() else 'cpu')
        dtype = torch.float32 if device == 'cpu' else torch.float16
        if device == 'cuda' and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, local_files_only=True,
            **model_dtype_kwargs(transformers.__version__, dtype)).to(device).eval()
        print(f'Model runtime: transformers={transformers.__version__}, '
              f'tokenizer={type(self.tokenizer).__name__}, device={device}', flush=True)
        self.device = device
        limit = getattr(self.model.config, 'max_position_embeddings', args.context_length)
        self.context_length = min(args.context_length, limit)
        torch.manual_seed(args.seed)

    def generate(self, messages):
        result = self.generate_batch([messages])[0]
        if isinstance(result, Exception):
            raise result
        return result

    def generate_batch(self, histories):
        results = [None] * len(histories)
        valid = []
        for index, messages in enumerate(histories):
            try:
                ids = chat_input_ids(self.tokenizer, messages)
                if len(ids) + self.args.max_new_tokens > self.context_length:
                    raise ContextLimit(f'{len(ids)} prompt + {self.args.max_new_tokens} reserved output '
                        f'exceeds context limit {self.context_length}; history was not trimmed')
                valid.append((index, ids))
            except Exception as exc:
                results[index] = exc
        if not valid:
            return results
        t, args = self.tokenizer, self.args
        width = max(len(ids) for _, ids in valid)
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = t.eos_token_id
        eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos]) - {None}
        pad = t.pad_token_id if t.pad_token_id is not None else t.eos_token_id
        if pad is None:
            raise ValueError('Tokenizer needs a pad or EOS token for batched generation')
        inputs = self.torch.tensor([[pad] * (width-len(ids)) + ids for _, ids in valid],
            dtype=self.torch.long, device=self.device)
        mask = self.torch.tensor([[0] * (width-len(ids)) + [1] * len(ids) for _, ids in valid],
            dtype=self.torch.long, device=self.device)
        options = dict(max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0, pad_token_id=pad)
        if args.temperature > 0:
            options.update(temperature=args.temperature, top_p=args.top_p, top_k=0)
        with self.torch.inference_mode():
            output = self.model.generate(input_ids=inputs,
                attention_mask=mask, **options)
        completions = output[:, width:].tolist()
        for (index, ids), completion in zip(valid, completions):
            ended = False
            for pos, token in enumerate(completion):
                if token in eos_ids:
                    completion = completion[:pos+1]
                    ended = True
                    break
            results[index] = (t.decode(completion, skip_special_tokens=True), {
                'prompt_token_ids': ids, 'completion_token_ids': completion,
                'prompt_tokens': len(ids), 'completion_tokens': len(completion),
                'inference_batch_size': len(valid),
                'output_limit_reached': not ended and len(completion) >= args.max_new_tokens,
            })
        return results


class LocalAPIBackend:
    def __init__(self, args):
        self.args = args
        url = urlparse(args.base_url)
        if url.scheme not in ('http', 'https'):
            raise ValueError('Local server URL must use http or https')
        try:
            local = ipaddress.ip_address(url.hostname).is_loopback
        except ValueError:
            local = url.hostname == 'localhost'
        if not local:
            raise ValueError('--base-url must point to a loopback address (localhost/127.0.0.1)')

    def generate(self, messages):
        args = self.args
        body = dict(model=args.served_model_name, messages=messages,
            temperature=args.temperature, top_p=args.top_p,
            max_tokens=args.max_new_tokens, seed=args.seed,
            chat_template_kwargs={'enable_thinking': False})
        req = Request(args.base_url.rstrip('/') + '/chat/completions',
            data=json.dumps(body).encode(), headers={
                'Content-Type': 'application/json', 'Authorization': 'Bearer ' + args.api_key})
        with urlopen(req, timeout=args.model_timeout) as response:
            result = json.load(response)
        choice = result['choices'][0]
        content = choice['message'].get('content')
        if not content:
            raise RuntimeError('Local server returned empty content; check non-thinking configuration')
        return content, {'usage': result.get('usage'), 'finish_reason': choice.get('finish_reason')}


def run_episode(args, backend, task_id, run_name, out_dir):
    worker = None
    result = dict(task_id=task_id, experiment_name=run_name,
        success=False, evaluated=False, declared_done=False, steps=0,
        executed_steps=0, format_errors=0, execution_errors=0, authentication_errors=0,
        completion_tokens=0, model_wait_seconds=0., environment_seconds=0.,
        termination='error', error=None)
    path = out_dir / f'{task_id}.jsonl'
    started = time.monotonic()
    try:
        worker = WorldProcess(args.appworld_python, args.appworld_root, args.env_timeout)
        task = worker.call('reset', task_id=task_id, experiment_name=run_name,
            max_interactions=args.max_steps, seed=args.seed)
        messages = prompt_messages(task['supervisor'], task['instruction'])
        with path.open('w') as log:
            def record(row):
                log.write(json.dumps(row, ensure_ascii=False) + '\n')
                log.flush()
            record(dict(type='initial', task=task, messages=messages))
            for step in range(1, args.max_steps + 1):
                try:
                    generate_started = time.monotonic()
                    raw, generation = backend.generate(messages)
                except ContextLimit as exc:
                    result.update(termination='context_limit', error=str(exc))
                    break
                result['model_wait_seconds'] += time.monotonic() - generate_started
                result['completion_tokens'] += generation.get('completion_tokens',
                    (generation.get('usage') or {}).get('completion_tokens', 0))
                messages.append(dict(role='assistant', content=raw))
                try:
                    code = extract_code(raw)
                except (ValueError, SyntaxError) as exc:
                    code = None
                    result['format_errors'] += 1
                    observation = format_error_feedback(exc)
                    raw_observation = None
                    feedback_kind = 'format_error'
                    declared_done = False
                else:
                    execute_started = time.monotonic()
                    execution = worker.call('execute', code=code)
                    result['environment_seconds'] += time.monotonic() - execute_started
                    result['executed_steps'] += 1
                    raw_observation = execution['observation']
                    observation, execution_failed = execution_feedback(raw_observation)
                    feedback_kind = 'execution_error' if execution_failed else 'execution_output'
                    result['execution_errors'] += int(execution_failed)
                    result['authentication_errors'] += int(bool(re.search(
                        r'Response status code is\s+(401|403)\b', raw_observation)))
                    declared_done = execution['declared_done']
                if getattr(args, 'observation_tokens', 0):
                    from observation_filter import bound_observation
                    owner = getattr(backend, 'backend', backend)
                    observation, _ = bound_observation(owner.observation_tokenizer, observation, args.observation_tokens)
                result['steps'] = step
                record(dict(type='turn', step=step, raw_completion=raw,
                    executed_code=code, observation=observation,
                    raw_observation=raw_observation, feedback_kind=feedback_kind,
                    declared_done=declared_done, generation=generation))
                print(f'{task_id} turn {step}/{args.max_steps}: '
                    f'{"declared done" if declared_done else "continuing"}', flush=True)
                if args.verbose:
                    print(raw, '\nOBSERVATION:\n', observation, flush=True)
                if declared_done:
                    result.update(declared_done=True, termination='declared_done')
                    break
                messages.append(dict(role='user', content=observation))
            else:
                result['termination'] = 'step_limit'
            # Evaluator feedback never goes back into the model's history.
            evaluation = worker.call('evaluate')
            result.update(evaluated=True, success=evaluation['success'],
                passed_checks=evaluation['passed_checks'], total_checks=evaluation['total_checks'])
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['traceback'] = traceback.format_exc()
        result['termination'] = 'error'
    finally:
        # Normal closure lets AppWorld persist/close its own state before exit.
        if worker is not None and worker.proc.poll() is None:
            try:
                worker.call('close')
            except Exception:
                pass
        if worker is not None:
            worker.close()
    result['seconds'] = round(time.monotonic() - started, 3)
    # Write terminal status even when a model/worker error interrupts the loop.
    with path.open('a') as stream:
        stream.write(json.dumps(dict(type='result', **result), ensure_ascii=False) + '\n')
    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['transformers', 'api'], default='transformers')
    p.add_argument('--model', help='Downloaded Qwen3-4B directory (Transformers backend)')
    p.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--served-model-name', default='qwen3-4b')
    p.add_argument('--api-key', default='local')
    p.add_argument('--appworld-python', default=sys.executable)
    p.add_argument('--appworld-root', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'dev', 'test_normal', 'test_challenge'], default='train')
    p.add_argument('--task-id', action='append', help='Repeat to select tasks within --split')
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--limit', type=int, default=1, help='Default: one task; 0 means all selected tasks')
    selection.add_argument('--all-tasks', action='store_true', help='Run every task in the selected split once')
    p.add_argument('--max-steps', type=int, default=50)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--context-length', type=int, default=32768)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--env-timeout', type=float, default=180)
    p.add_argument('--model-timeout', type=float, default=300)
    p.add_argument('--output-dir', type=Path, default=HERE / 'runs')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--observation-tokens', type=int, default=0,
                   help='Optional RL-matched observation token cap; 0 disables, requires --model tokenizer')
    p.add_argument('--concurrency', type=int, default=1,
        help='Independent task/World subprocesses in flight; model is loaded once')
    p.add_argument('--inference-batch-size', type=int, default=4,
        help='Maximum Transformers microbatch size across concurrent tasks')
    p.add_argument('--batch-wait-ms', type=float, default=20,
        help='Maximum inference queue collection delay in milliseconds')
    p.add_argument('--check-env', action='store_true',
        help='Reset one task, execute a smoke check, evaluate and exit without loading the LLM')
    a = p.parse_args()
    if a.all_tasks:
        a.limit = 0
    if a.backend == 'transformers' and not a.model and not a.check_env:
        p.error('--model is required for Transformers')
    if min(a.max_steps, a.max_new_tokens, a.context_length, a.env_timeout, a.model_timeout) <= 0:
        p.error('Budgets and timeouts must be positive')
    if (a.limit < 0 or a.temperature < 0 or not 0 < a.top_p <= 1
            or a.concurrency < 1 or a.inference_batch_size < 1 or a.batch_wait_ms < 0):
        p.error('Invalid limit or sampling parameters')
    if a.observation_tokens and (a.observation_tokens < 128 or not a.model):
        p.error('--observation-tokens requires a local --model tokenizer and a cap >= 128')
    a.appworld_root = a.appworld_root.expanduser().resolve()
    if not a.appworld_root.is_dir():
        p.error('--appworld-root must be an existing directory with downloaded data')
    return a


def main():
    args = parse_args()
    probe = WorldProcess(args.appworld_python, args.appworld_root, args.env_timeout)
    try:
        available = probe.call('list', split=args.split)
    finally:
        probe.close()
    tasks = list(dict.fromkeys(args.task_id or available))
    if any(task not in available for task in tasks):
        raise ValueError('Requested task ID is not in the selected split')
    tasks = tasks[:args.limit] if args.limit else tasks
    if not tasks:
        raise ValueError('No tasks selected')
    if args.check_env:
        worker = WorldProcess(args.appworld_python, args.appworld_root, args.env_timeout)
        try:
            task = worker.call('reset', task_id=tasks[0],
                experiment_name=f'qwen3_env_check/{uuid.uuid4().hex}',
                max_interactions=args.max_steps, seed=args.seed)
            execution = worker.call('execute', code='print(1 + 1)')
            evaluation = worker.call('evaluate')
            print(json.dumps(dict(runtime=task['runtime'], execution=execution,
                evaluation=evaluation), indent=2, ensure_ascii=False))
            worker.call('close')
        finally:
            worker.close()
        return 0
    run_name = f'qwen3_local/{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}'
    out = args.output_dir.expanduser().resolve() / run_name.replace('/', '_')
    out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.pop('api_key')
    (out / 'config.json').write_text(json.dumps(dict(config=config, tasks=tasks,
        upstream=json.loads((HERE / 'upstream.json').read_text())), indent=2) + '\n')
    backend = TransformersBackend(args) if args.backend == 'transformers' else LocalAPIBackend(args)
    if args.observation_tokens:
        if hasattr(backend, 'tokenizer'):
            backend.observation_tokenizer = backend.tokenizer
        else:
            from transformers import AutoTokenizer
            backend.observation_tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.backend == 'transformers' and args.concurrency > 1:
        backend = BatchingBackend(backend, args.inference_batch_size, args.batch_wait_ms)
    results = []
    save_results(out, results, len(tasks))
    started = time.monotonic()
    order = {task: index for index, task in enumerate(tasks)}
    print(f'Evaluating {len(tasks)} tasks, concurrency={args.concurrency}; model loaded once', flush=True)
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(run_episode, args, backend, task, run_name, out) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                results.sort(key=lambda row: order[row['task_id']])
                print(json.dumps(result, ensure_ascii=False), flush=True)
                metrics = backend.metrics() if isinstance(backend, BatchingBackend) else None
                summary = save_results(out, results, len(tasks), time.monotonic()-started, metrics)
                index = len(results)
                print(f'Progress {index}/{len(tasks)} | success {summary["successes"]}/{index} '
                      f'({summary["success_rate"]:.1%}) | episodes/min {summary["episodes_per_minute"]:.2f} '
                      f'| format errors {summary["format_errors"]} '
                      f'| execution errors {summary["execution_errors"]} '
                      f'| run errors {summary["errors"]}', flush=True)
    finally:
        if isinstance(backend, BatchingBackend):
            backend.close()
    print(f'Results: {out}')
    return 1 if any(r['termination'] == 'error' for r in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
