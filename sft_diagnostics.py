"""Local-only SFT length and completion-action diagnostics; no training changes."""
import ast
from collections import Counter
import math


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(count=0, min=None, max=None, mean=None, percentiles={}, histogram={})
    percentiles = {}
    for p in [50, 75, 90, 95, 99]:
        position = (len(values)-1)*p/100
        lo, hi = math.floor(position), math.ceil(position)
        percentiles[f'p{p}'] = values[lo] + (values[hi]-values[lo])*(position-lo)
    histogram, lower = {}, 0
    for upper in [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1000000]:
        histogram[f'{lower}-{upper}'] = sum(lower <= v <= upper for v in values)
        lower = upper+1
    histogram[f'{lower}+'] = sum(v >= lower for v in values)
    return dict(count=len(values), min=values[0], max=values[-1],
                mean=sum(values)/len(values), percentiles=percentiles, histogram=histogram)


def complete_task_actions(code):
    """Count syntactic calls to apis.supervisor.complete_task, not comments/strings.

    This does not prove runtime execution and does not resolve aliases.
    """
    actions = Counter()
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == 'complete_task'
                and isinstance(f.value, ast.Attribute) and f.value.attr == 'supervisor'
                and isinstance(f.value.value, ast.Name) and f.value.value.id == 'apis'):
            continue
        status = next((kw.value for kw in node.keywords if kw.arg == 'status'), None)
        dynamic_kwargs = any(kw.arg is None for kw in node.keywords)
        if status is None and not dynamic_kwargs:
            category = 'default_status'
        elif isinstance(status, ast.Constant) and status.value == 'fail':
            category = 'explicit_fail'
        else:
            category = 'other_or_dynamic_status'
        actions[category] += 1
    return actions


def rate(kept, total):
    return kept/total if total else None


class Diagnostics:
    def __init__(self):
        self.records = []
        self.by_task = {}
        self.action_counts = Counter()
        self.kept_action_counts = Counter()

    def add(self, row, line, lengths, kept):
        actions = complete_task_actions(row['completion'])
        record = dict(source_line=line, task_id=row['task_id'], step=row['step'],
                      **lengths, retained=kept, has_complete_task=bool(actions),
                      complete_task_call_count=sum(actions.values()))
        self.records.append(record)
        task = self.by_task.setdefault(row['task_id'], dict(task_id=row['task_id'],
            samples=0, retained=0, complete_task_samples=0, retained_complete_task_samples=0,
            complete_task_calls=0, retained_complete_task_calls=0, max_total_tokens=0))
        task['samples'] += 1
        task['retained'] += int(kept)
        task['complete_task_samples'] += int(bool(actions))
        task['retained_complete_task_samples'] += int(bool(actions) and kept)
        task['complete_task_calls'] += sum(actions.values())
        task['retained_complete_task_calls'] += sum(actions.values()) if kept else 0
        task['max_total_tokens'] = max(task['max_total_tokens'], lengths['total_tokens'])
        self.action_counts.update(actions)
        if kept:
            self.kept_action_counts.update(actions)
        return record

    def task_rows(self):
        return [dict(task, retention_rate=rate(task['retained'], task['samples']),
            complete_task_retention_rate=rate(task['retained_complete_task_samples'],
                                              task['complete_task_samples']))
                for _, task in sorted(self.by_task.items())]

    def summary(self):
        complete = [r for r in self.records if r['has_complete_task']]
        tasks = list(self.by_task.values())
        return dict(length_distribution={group: {key: distribution([
            r[key] for r in self.records if group == 'all' or r['retained']])
            for key in ['total_tokens', 'prompt_tokens', 'completion_tokens']}
            for group in ['all', 'retained']},
            tasks=dict(input=len(tasks), with_retained_samples=sum(t['retained'] > 0 for t in tasks),
                fully_retained=sum(t['samples'] == t['retained'] for t in tasks),
                partially_retained=sum(0 < t['retained'] < t['samples'] for t in tasks),
                fully_dropped=sum(t['retained'] == 0 for t in tasks)),
            complete_task=dict(definition='AST calls to apis.supervisor.complete_task in target code; aliases not resolved; not proof of execution or success',
                samples=len(complete), retained_samples=sum(r['retained'] for r in complete),
                sample_retention_rate=rate(sum(r['retained'] for r in complete), len(complete)),
                calls=sum(self.action_counts.values()), retained_calls=sum(self.kept_action_counts.values()),
                call_retention_rate=rate(sum(self.kept_action_counts.values()), sum(self.action_counts.values())),
                call_categories=dict(self.action_counts), retained_call_categories=dict(self.kept_action_counts),
                tasks_with_action=sum(t['complete_task_samples'] > 0 for t in tasks),
                tasks_with_retained_action=sum(t['retained_complete_task_samples'] > 0 for t in tasks),
                task_retention_rate=rate(sum(t['retained_complete_task_samples'] > 0 for t in tasks),
                                         sum(t['complete_task_samples'] > 0 for t in tasks))),
            longest_samples=sorted(self.records, key=lambda r:r['total_tokens'], reverse=True)[:20])


def describe_anomaly(tokenizer, row, record, preview_chars):
    from run_agent import normalize_input_ids
    messages = []
    previous_assistant = None
    for index, message in enumerate(row['messages']):
        content = message['content']
        tokens = (normalize_input_ids(tokenizer(content, add_special_tokens=False,
                                               return_attention_mask=False)) if content else [])
        item = dict(message_index=index, role=message['role'], characters=len(content),
                    content_tokens=len(tokens), preceding_assistant_index=previous_assistant)
        if preview_chars:
            item['preview'] = content[:preview_chars]
        messages.append(item)
        if message['role'] == 'assistant':
            previous_assistant = index
    return dict(record, source_sha256=row.get('source_sha256'),
        note='Message token counts encode content independently, exclude chat-template overhead and are not additive. Indices are zero-based; source_line is one-based.',
        largest_messages=sorted(messages, key=lambda m:m['content_tokens'], reverse=True)[:5],
        messages=messages)
