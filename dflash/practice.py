"""Single-prompt Colab helpers, independent of the benchmark module."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model import DFlashDraftModel


def _prepare_fp16_context_projection(draft):
    # The BF16-trained checkpoint can overflow FP16 in fc, before RMSNorm.
    # Keep just this projection and its normalization in FP32, then return
    # FP16 context to the unmodified decoder. Parameter names/shapes stay intact.
    draft.fc.float()
    draft.fc.register_forward_pre_hook(lambda module, args: (args[0].float(),))
    draft.hidden_norm.register_forward_hook(
        lambda module, args, output: output.to(torch.float16)
    )


def load_practice_models(
    target_id="Qwen/Qwen3-4B",
    draft_id="z-lab/Qwen3-4B-DFlash-b16",
    device="cuda:0",
    dtype=None,
):
    """Return (target, draft, target_tokenizer), using SDPA on one CUDA device.

    No student implementation is needed for loading. By default, choose BF16
    on supported GPUs and FP16 on GPUs such as the Colab T4. For FP16, the
    context projection/normalization uses FP32 to avoid checkpoint overflow.
    """
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime before loading the practice models.")
    with torch.cuda.device(device):
        if dtype is None:
            # Some PyTorch versions count BF16 emulation as support on T4.
            native_bf16 = (
                torch.cuda.is_bf16_supported()
                and torch.cuda.get_device_capability(device)[0] >= 8
            )
            dtype = torch.bfloat16 if native_bf16 else torch.float16
        target = AutoModelForCausalLM.from_pretrained(
            target_id, attn_implementation="sdpa", dtype=dtype,
        ).to(device).eval()
        draft = DFlashDraftModel.from_pretrained(
            draft_id, attn_implementation="sdpa", dtype=dtype,
        ).to(device).eval()
        if dtype == torch.float16:
            _prepare_fp16_context_projection(draft)
    return target, draft, AutoTokenizer.from_pretrained(target_id)


def stop_token_ids(model, tokenizer):
    """Use the target generation configuration, falling back to its tokenizer."""
    token_ids = model.generation_config.eos_token_id
    if token_ids is None:
        token_ids = tokenizer.eos_token_id
    if token_ids is None:
        return []
    return [token_ids] if isinstance(token_ids, int) else list(token_ids)


def apply_practice_chat_template(tokenizer, messages, *, enable_thinking=False):
    """Render Qwen3 chat text; the practice defaults to non-thinking mode."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
