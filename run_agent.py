"""Local Qwen3 ReAct runner based on AppWorld's minimal_agent.ipynb."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Mapping
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


def save_results(out, results, planned):
    summary = build_summary(results, planned)
    temp = out / 'summary.json.tmp'
    temp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    temp.replace(out / 'summary.json')
    fields = ['task_id', 'success', 'evaluated', 'declared_done', 'termination', 'steps',
              'executed_steps', 'format_errors', 'execution_errors', 'authentication_errors',
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
        t, args = self.tokenizer, self.args
        ids = chat_input_ids(t, messages)
        if len(ids) + args.max_new_tokens > self.context_length:
            raise ContextLimit(f'{len(ids)} prompt + {args.max_new_tokens} reserved output '
                f'exceeds context limit {self.context_length}; history was not trimmed')
        inputs = self.torch.tensor([ids], dtype=self.torch.long, device=self.device)
        options = dict(max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0, pad_token_id=t.eos_token_id)
        if args.temperature > 0:
            options.update(temperature=args.temperature, top_p=args.top_p, top_k=0)
        with self.torch.inference_mode():
            output = self.model.generate(input_ids=inputs,
                attention_mask=self.torch.ones_like(inputs), **options)
        completion = output[0, len(ids):].tolist()
        return t.decode(completion, skip_special_tokens=True), {
            'prompt_token_ids': ids, 'completion_token_ids': completion,
            'prompt_tokens': len(ids), 'completion_tokens': len(completion),
            'output_limit_reached': len(completion) >= args.max_new_tokens,
        }


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
    worker = WorldProcess(args.appworld_python, args.appworld_root, args.env_timeout)
    result = dict(task_id=task_id, experiment_name=run_name,
        success=False, evaluated=False, declared_done=False, steps=0,
        executed_steps=0, format_errors=0, execution_errors=0, authentication_errors=0,
        termination='error', error=None)
    path = out_dir / f'{task_id}.jsonl'
    started = time.monotonic()
    try:
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
                    raw, generation = backend.generate(messages)
                except ContextLimit as exc:
                    result.update(termination='context_limit', error=str(exc))
                    break
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
                    execution = worker.call('execute', code=code)
                    result['executed_steps'] += 1
                    raw_observation = execution['observation']
                    observation, execution_failed = execution_feedback(raw_observation)
                    feedback_kind = 'execution_error' if execution_failed else 'execution_output'
                    result['execution_errors'] += int(execution_failed)
                    result['authentication_errors'] += int(bool(re.search(
                        r'Response status code is\s+(401|403)\b', raw_observation)))
                    declared_done = execution['declared_done']
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
        if worker.proc.poll() is None:
            try:
                worker.call('close')
            except Exception:
                pass
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
    p.add_argument('--check-env', action='store_true',
        help='Reset one task, execute a smoke check, evaluate and exit without loading the LLM')
    a = p.parse_args()
    if a.all_tasks:
        a.limit = 0
    if a.backend == 'transformers' and not a.model and not a.check_env:
        p.error('--model is required for Transformers')
    if min(a.max_steps, a.max_new_tokens, a.context_length, a.env_timeout, a.model_timeout) <= 0:
        p.error('Budgets and timeouts must be positive')
    if a.limit < 0 or a.temperature < 0 or not 0 < a.top_p <= 1:
        p.error('Invalid limit or sampling parameters')
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
    results = []
    save_results(out, results, len(tasks))
    for index, task in enumerate(tasks, 1):
        print(f'\n=== Task {index}/{len(tasks)}: {task} ===', flush=True)
        result = run_episode(args, backend, task, run_name, out)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        summary = save_results(out, results, len(tasks))
        print(f'Progress {index}/{len(tasks)} | success {summary["successes"]}/{index} '
              f'({summary["success_rate"]:.1%}) | format errors {summary["format_errors"]} '
              f'| execution errors {summary["execution_errors"]} '
              f'| run errors {summary["errors"]}', flush=True)
    print(f'Results: {out}')
    return 1 if any(r['termination'] == 'error' for r in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
