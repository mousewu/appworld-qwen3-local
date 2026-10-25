"""Completion-only AppWorld SFT adapter for THUDM/slime's custom rollout API."""
import hashlib
import json
from pathlib import Path

SLIME_REVISION = '18868a79b0b083a093798a94ed0f8cb4f71b9c14'


def tokenizer_signature(model):
    digest = hashlib.sha256()
    found = False
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                 'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt']:
        path = Path(model) / name
        if path.is_file():
            found = True
            digest.update(name.encode())
            digest.update(path.read_bytes())
    if not found:
        raise ValueError('No tokenizer/config files found in model directory')
    return digest.hexdigest()


def prepare_row(tokenizer, row, max_length, signature, diagnostics=None):
    from run_agent import extract_code
    from train_lora import encode_sample
    code = row['completion']
    if extract_code(code) != code.strip():
        raise ValueError('Completion must be raw Python, matching the inference prompt')
    encoded = encode_sample(tokenizer, row, max_length, diagnostics=diagnostics)
    if encoded is None:
        return None
    labels = encoded['labels']
    first = next(i for i, value in enumerate(labels) if value != -100)
    if first < 1 or labels[first:] != encoded['input_ids'][first:]:
        raise ValueError('Expected exactly one supervised completion suffix')
    return dict(prompt=f"AppWorld Train {row['task_id']} turn {row['step']}", metadata=dict(
        source_name='appworld_train', task_id=row['task_id'], step=row['step'],
        tokens=encoded['input_ids'], response_length=len(labels)-first,
        tokenizer_signature=signature, schema='appworld_completion_v1'))


def populate_sample(sample, signature):
    metadata = sample.metadata
    if metadata.get('schema') != 'appworld_completion_v1':
        raise ValueError('Use prepare_slime_sft.py to build the input dataset')
    if metadata.get('tokenizer_signature') != signature:
        raise ValueError('Tokenizer/config differs from preprocessing; regenerate data')
    tokens = metadata['tokens']
    length = metadata['response_length']
    if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError('Invalid token IDs')
    if type(length) is not int or not 0 < length < len(tokens):
        raise ValueError('Invalid completion length')
    sample.tokens = tokens.copy()
    sample.response_length = length
    sample.loss_mask = [1] * length
    sample.reward = 0
    return sample


_SIGNATURE_CACHE = {}


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Same Sample contract as slime.rollout.sft_rollout.generate_rollout.

    Slime left-pads the response mask with zeros for the prompt. Keeping only
    the final response here masks all few-shot and historical assistant turns.
    """
    if evaluation:
        raise ValueError('Offline SFT adapter does not implement environment evaluation')
    key = str(Path(args.hf_checkpoint).resolve())
    if key not in _SIGNATURE_CACHE:
        _SIGNATURE_CACHE[key] = tokenizer_signature(key)
    samples = data_buffer.get_samples(args.rollout_batch_size)
    for group in samples:
        if len(group) != 1:
            raise ValueError('SFT requires --n-samples-per-prompt 1')
        populate_sample(group[0], _SIGNATURE_CACHE[key])
    return samples
