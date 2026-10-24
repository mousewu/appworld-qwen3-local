"""Local Qwen3 LoRA SFT: next-code loss only, no history/observation loss."""
import argparse
import json
from pathlib import Path

from run_agent import chat_input_ids, normalize_input_ids, model_dtype_kwargs


def encode_sample(tokenizer, row, max_length):
    prefix = chat_input_ids(tokenizer, row['messages'])
    # Render the answer through the same chat template, including end-of-turn.
    full = normalize_input_ids(tokenizer(tokenizer.apply_chat_template(
        row['messages'] + [dict(role='assistant', content=row['completion'])],
        tokenize=False, add_generation_prompt=False, enable_thinking=False),
        add_special_tokens=False, return_attention_mask=False))
    if full[:len(prefix)] != prefix:
        raise ValueError('Chat template prompt is not a token prefix of completed chat')
    if len(full) > max_length:
        return None  # Never truncate away task context, code, or end-of-turn.
    if len(full) <= len(prefix):
        raise ValueError('Empty supervised completion')
    return dict(input_ids=full, attention_mask=[1] * len(full),
                labels=[-100] * len(prefix) + full[len(prefix):])


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        width = max(len(f['input_ids']) for f in features)
        return {key: torch.tensor([f[key] + [pad] * (width-len(f[key])) for f in features])
                for key, pad in [('input_ids', self.pad_token_id),
                                 ('attention_mask', 0), ('labels', -100)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--max-length', type=int, default=8192)
    p.add_argument('--epochs', type=float, default=1)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--gradient-accumulation-steps', type=int, default=16)
    p.add_argument('--lora-r', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--merge-output', type=Path, help='Optional standalone model for run_agent.py')
    a = p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error('Output directory must be empty')
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error('This training entry point requires a CUDA GPU with BF16 support')
    set_seed(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded, skipped = [], 0
    with a.data.open() as stream:
        for line in stream:
            item = encode_sample(tokenizer, json.loads(line), a.max_length)
            if item is None:
                skipped += 1
            else:
                encoded.append(item)
    if not encoded:
        p.error('No samples fit max-length')
    print(f'Training samples: {len(encoded)}; oversized samples skipped: {skipped}')
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
        **model_dtype_kwargs(transformers.__version__, torch.bfloat16))
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=a.lora_r,
        lora_alpha=2*a.lora_r, lora_dropout=0.05, bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    model.print_trainable_parameters()
    args = TrainingArguments(output_dir=str(a.output_dir), num_train_epochs=a.epochs,
        learning_rate=a.learning_rate, per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=1, save_strategy='epoch', save_total_limit=2, report_to='none',
        seed=a.seed, optim='adamw_torch', remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(encoded),
                      data_collator=CompletionCollator(tokenizer.pad_token_id))
    trainer.train()
    trainer.save_model(str(a.output_dir))
    tokenizer.save_pretrained(a.output_dir)
    (a.output_dir / 'data_stats.json').write_text(json.dumps(dict(
        samples=len(encoded), skipped_overlength=skipped, max_length=a.max_length), indent=2))
    if a.merge_output:
        if a.merge_output.exists() and any(a.merge_output.iterdir()):
            raise ValueError('Merge output directory must be empty')
        model = model.merge_and_unload()
        model.save_pretrained(a.merge_output, safe_serialization=True)
        tokenizer.save_pretrained(a.merge_output)


if __name__ == '__main__':
    main()
