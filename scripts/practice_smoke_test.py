"""Instructor smoke test using the real Qwen3-4B and DFlash-b16 checkpoints.

Run from the checkout: python scripts/practice_smoke_test.py
This script contains a reference answer; the student module remains a stub.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import transformers

import dflash.student_block as student_block
from dflash.model import dflash_generate
from dflash.practice import (
    apply_practice_chat_template,
    load_practice_models,
    stop_token_ids,
)


def reference_parallel_block_draft(
    model, target_hidden, noise_embedding, draft_position_ids,
    past_key_values_draft, output_head, verify_size,
):
    draft_hidden = model(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=draft_position_ids,
        past_key_values=past_key_values_draft,
        use_cache=True,
    )[:, 1 - verify_size :, :]
    draft_logits = model.compute_logits(draft_hidden, output_head)
    draft_tokens = torch.argmax(draft_logits, dim=-1)
    assert draft_tokens.shape == (1, verify_size - 1)
    assert draft_tokens.dtype == torch.long
    return draft_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[16])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16"], default="auto")
    parser.add_argument("--prompt", default="Explain in one sentence why the sky is blue.")
    args = parser.parse_args()
    if args.max_new_tokens < 2 or any(size < 2 or size > 16 for size in args.block_sizes):
        parser.error("Use at least 2 new tokens and block sizes from 2 through 16.")
    if not torch.cuda.is_available():
        parser.exit(1, "CUDA GPU required; no models were loaded.\n")

    print(f"torch={torch.__version__}, transformers={transformers.__version__}", flush=True)
    dtype = None if args.dtype == "auto" else getattr(torch, args.dtype)
    # Intentionally load before replacing the student stub.
    target, draft, tokenizer = load_practice_models(device=args.device, dtype=dtype)
    print(f"GPU={torch.cuda.get_device_name(target.device)}, dtype={target.dtype}", flush=True)
    text = apply_practice_chat_template(tokenizer, [{"role": "user", "content": args.prompt}])
    input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(target.device)
    eos_ids = stop_token_ids(target, tokenizer)
    original_hook = student_block.parallel_block_draft
    calls = 0

    def counted_reference(**kwargs):
        nonlocal calls
        calls += 1
        return reference_parallel_block_draft(**kwargs)

    student_block.parallel_block_draft = counted_reference
    try:
        for block_size in args.block_sizes:
            # Warm up each shape before collecting short-run timing statistics.
            finite_checks = []
            handle = draft.register_forward_hook(
                lambda module, inputs, output: finite_checks.append(torch.isfinite(output).all())
            )
            try:
                dflash_generate(draft, target, input_ids, args.max_new_tokens, eos_ids, block_size=block_size)
            finally:
                handle.remove()
            assert finite_checks, "Prompt stopped before reaching the student hook."
            assert torch.stack(finite_checks).all(), "Non-finite drafter hidden states."
            calls = 0
            result = dflash_generate(
                draft, target, input_ids, args.max_new_tokens, eos_ids,
                temperature=0.0, block_size=block_size, return_stats=True,
            )
            assert calls > 0, "Prompt stopped before reaching the student hook."
            assert 0 < result.num_output_tokens <= args.max_new_tokens
            assert calls == len(result.acceptance_lengths)
            print(json.dumps({
                "block_size": block_size,
                "student_hook_calls": calls,
                "acceptance_lengths": result.acceptance_lengths,
                "average_acceptance_length": statistics.mean(result.acceptance_lengths),
                "num_output_tokens": result.num_output_tokens,
                "time_per_output_token": result.time_per_output_token,
                "tps": 1 / result.time_per_output_token,
                "text": tokenizer.decode(result.output_ids[0, result.num_input_tokens:], skip_special_tokens=True),
            }, ensure_ascii=False), flush=True)
    finally:
        student_block.parallel_block_draft = original_hook


if __name__ == "__main__":
    main()
