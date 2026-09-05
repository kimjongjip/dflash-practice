"""Loader checks without downloading models or requiring a CUDA device."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from dflash import practice


class PracticeTests(unittest.TestCase):
    def test_fp16_projection_does_not_overflow_before_normalization(self):
        draft = SimpleNamespace(
            fc=torch.nn.Linear(2, 1, bias=False).half(),
            hidden_norm=Qwen3RMSNorm(1, eps=1e-6).half(),
        )
        with torch.no_grad():
            draft.fc.weight.fill_(10)
            hidden = torch.tensor([[[40000., 40000.]]], dtype=torch.float16)
            self.assertFalse(torch.isfinite(draft.hidden_norm(draft.fc(hidden))).all())
            practice._prepare_fp16_context_projection(draft)
            normalized = draft.hidden_norm(draft.fc(hidden))
        self.assertEqual(draft.fc.weight.dtype, torch.float32)
        self.assertEqual(normalized.dtype, torch.float16)
        torch.testing.assert_close(normalized, torch.ones_like(normalized))

    def test_loader_selects_native_dtype_and_shared_device(self):
        for supported, capability, expected in [
            (True, (8, 6), torch.bfloat16),
            (False, (7, 5), torch.float16),
            (True, (7, 5), torch.float16),  # T4 with BF16 emulation
        ]:
            with self.subTest(supported=supported, capability=capability):
                with (
                    patch.object(torch.cuda, "is_available", return_value=True),
                    patch.object(torch.cuda, "device", return_value=nullcontext()),
                    patch.object(torch.cuda, "is_bf16_supported", return_value=supported),
                    patch.object(torch.cuda, "get_device_capability", return_value=capability),
                    patch.object(practice.AutoModelForCausalLM, "from_pretrained") as load_target,
                    patch.object(practice.DFlashDraftModel, "from_pretrained") as load_draft,
                    patch.object(practice.AutoTokenizer, "from_pretrained") as load_tokenizer,
                ):
                    target, draft, tokenizer = practice.load_practice_models(device="cuda:1")
                for loader, model_id, result in [
                    (load_target, "Qwen/Qwen3-4B", target),
                    (load_draft, "z-lab/Qwen3-4B-DFlash-b16", draft),
                ]:
                    loader.assert_called_once_with(model_id, attn_implementation="sdpa", dtype=expected)
                    loader.return_value.to.assert_called_once_with(torch.device("cuda:1"))
                    loader.return_value.to.return_value.eval.assert_called_once_with()
                    self.assertIs(result, loader.return_value.to.return_value.eval.return_value)
                load_tokenizer.assert_called_once_with("Qwen/Qwen3-4B")
                self.assertIs(tokenizer, load_tokenizer.return_value)

    def test_stop_ids_include_zero_and_handle_absent_eos(self):
        for eos, fallback, expected in [(0, 7, [0]), ([2, 3], 7, [2, 3]), (None, 7, [7]), (None, None, [])]:
            model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=eos))
            tokenizer = SimpleNamespace(eos_token_id=fallback)
            self.assertEqual(practice.stop_token_ids(model, tokenizer), expected)

    def test_qwen_chat_defaults_to_non_thinking(self):
        tokenizer = Mock()
        messages = [{"role": "user", "content": "Hello"}]
        result = practice.apply_practice_chat_template(tokenizer, messages)
        tokenizer.apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        self.assertIs(result, tokenizer.apply_chat_template.return_value)


if __name__ == "__main__":
    unittest.main()
