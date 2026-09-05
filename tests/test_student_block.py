"""CPU-only hook contracts and dispatch checks; no checkpoints are downloaded."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers import Qwen3Config

import dflash.model as inference
from dflash import student_block


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
    return torch.argmax(draft_logits, dim=-1)


class StudentBlockContractTests(unittest.TestCase):
    def test_one_forward_and_parallel_argmax(self):
        hidden = torch.tensor([[[99., 0., 0.], [0., 2., 0.],
                                [0., 0., 3.], [4., 0., 0.]]])
        model = Mock(return_value=hidden)
        model.compute_logits.side_effect = lambda states, head: head(states)
        head = Mock(side_effect=lambda states: states.flip(-1))
        context, noise, positions, cache = object(), object(), object(), object()
        tokens = reference_parallel_block_draft(
            model, context, noise, positions, cache, head, 4,
        )
        model.assert_called_once_with(
            target_hidden=context, noise_embedding=noise, position_ids=positions,
            past_key_values=cache, use_cache=True,
        )
        model.compute_logits.assert_called_once()
        head.assert_called_once()
        torch.testing.assert_close(head.call_args.args[0], hidden[:, -3:, :])
        self.assertEqual(tokens.shape, (1, 3))
        self.assertEqual(tokens.dtype, torch.long)
        self.assertEqual(tokens.tolist(), [[1, 0, 2]])

    def test_student_stub_has_no_solution(self):
        with self.assertRaisesRegex(NotImplementedError, "notebook"):
            student_block.parallel_block_draft(*(None,) * 7)


def fake_models(draft_class=inference.DFlashDraftModel):
    """Deterministic test doubles for orchestration, never used by the lesson."""
    config = Qwen3Config(
        vocab_size=32, hidden_size=32, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
        head_dim=32,
    )
    embedding = torch.nn.Embedding.from_pretrained(torch.eye(32))
    head = torch.nn.Identity()
    target = Mock(config=config, device=torch.device("cpu"), lm_head=head)
    target.get_input_embeddings.return_value = embedding

    def target_forward(input_ids, *, past_key_values, logits_to_keep=0, **kwargs):
        hidden = embedding(input_ids)
        kv = hidden.unsqueeze(1)
        past_key_values.update(kv, kv, 0)
        logits = torch.nn.functional.one_hot((input_ids + 1) % 31, 32).float() * 1000
        return SimpleNamespace(
            logits=logits[:, -logits_to_keep:, :], hidden_states=(hidden, hidden),
        )

    target.side_effect = target_forward
    draft = Mock(spec=draft_class)
    draft.config, draft.block_size, draft.mask_token_id = config, 4, 31
    draft.target_layer_ids = [0]

    def draft_forward(*, target_hidden, noise_embedding, past_key_values, **kwargs):
        kv = torch.cat([target_hidden, noise_embedding], dim=1).unsqueeze(1)
        past_key_values.update(kv, kv, 0)
        anchor = noise_embedding[:, :1].argmax(-1)
        tokens = (anchor + torch.arange(noise_embedding.shape[1])) % 31
        return torch.nn.functional.one_hot(tokens, 32).float() * 1000

    draft.side_effect = draft_forward
    draft.compute_logits.side_effect = lambda hidden, output_head: output_head(hidden)
    if draft_class is inference.DFlash2DraftModel:
        def propose(hidden, anchor, output_head, temperature):
            logits = output_head(hidden)
            tokens = logits.argmax(-1)
            probs = torch.softmax(logits, dim=-1) if temperature > 0 else None
            return tokens, None, probs
        draft.propose.side_effect = propose
    return target, draft


class GenerationHookTests(unittest.TestCase):
    def generate(self, draft, target, **kwargs):
        return inference.dflash_generate(
            draft, target, torch.tensor([[1, 2]]), max_new_tokens=10,
            stop_token_ids=None, **kwargs,
        )

    def test_stub_fails_at_drafting_after_prefill(self):
        target, draft = fake_models()
        with self.assertRaises(NotImplementedError):
            self.generate(draft, target)
        target.assert_called_once()
        draft.assert_not_called()

    def test_module_patch_sweep_and_cache_ownership(self):
        target, draft = fake_models()
        for block_size in [2, 4, 8, 12, 16]:
            with self.subTest(block_size=block_size):
                calls = []

                def student(**kwargs):
                    cache = kwargs["past_key_values_draft"]
                    context = kwargs["target_hidden"]
                    positions = kwargs["draft_position_ids"]
                    start = positions[0, -kwargs["verify_size"]].item()
                    self.assertEqual(cache.get_seq_length(), start - context.shape[1])
                    tokens = reference_parallel_block_draft(**kwargs)
                    self.assertEqual(cache.get_seq_length(), start + kwargs["verify_size"])
                    calls.append(kwargs["verify_size"])
                    return tokens

                draft.reset_mock()
                target.reset_mock()
                with patch.object(student_block, "parallel_block_draft", side_effect=student) as hook:
                    with patch.object(inference, "_cuda_time", side_effect=[0., 1., 2., 3.]):
                        result = self.generate(draft, target, block_size=block_size, return_stats=True)
                self.assertEqual(result.output_ids.tolist(), [list(range(1, 13))])
                self.assertEqual(result.num_output_tokens, 10)
                self.assertEqual(sum(result.acceptance_lengths), 9)
                self.assertEqual(result.time_per_output_token, 0.1)
                self.assertEqual(hook.call_count, len(calls))
                self.assertEqual(draft.call_count, hook.call_count)
                self.assertEqual(target.call_count, hook.call_count + 1)
                self.assertTrue(all(1 < size <= block_size for size in calls))

    def test_verification_rejects_wrong_drafts(self):
        target, draft = fake_models()

        def wrong_draft(**kwargs):
            return (reference_parallel_block_draft(**kwargs) + 7) % 31

        with patch.object(student_block, "parallel_block_draft", side_effect=wrong_draft):
            result = self.generate(draft, target)
        self.assertEqual(result.tolist(), [list(range(1, 13))])
        self.assertEqual(draft.call_count, 9)

    def test_other_paths_do_not_call_student_hook(self):
        paths = [
            (inference.DFlashDraftModel, 0., 1),
            (inference.DFlashDraftModel, 0.7, 4),
            (inference.DFlash2DraftModel, 0., 4),
            (inference.DFlash2DraftModel, 0.7, 4),
        ]
        for draft_class, temperature, block_size in paths:
            with self.subTest(model=draft_class.__name__, temperature=temperature, block_size=block_size):
                target, draft = fake_models(draft_class)
                with patch.object(student_block, "parallel_block_draft", side_effect=AssertionError("unexpected hook")) as hook:
                    result = self.generate(draft, target, temperature=temperature, block_size=block_size)
                hook.assert_not_called()
                self.assertEqual(result.tolist(), [list(range(1, 13))])
                if draft_class is inference.DFlash2DraftModel:
                    self.assertEqual(draft.propose.call_count, draft.call_count)
                    self.assertGreater(draft.propose.call_count, 0)


class CacheCompatibilityTests(unittest.TestCase):
    def test_real_dynamic_cache_preserves_equal_length_and_crops_suffix(self):
        cache = inference._make_cache(Qwen3Config(num_hidden_layers=1))
        kv = torch.ones(1, 1, 6, 2)
        cache.update(kv, kv, 0)
        inference._crop_to(cache, 6)
        self.assertEqual(cache.get_seq_length(), 6)
        inference._crop_to(cache, 3)
        self.assertEqual(cache.get_seq_length(), 3)

    def test_recording_api_remains_active_for_modern_cache(self):
        cache = Mock()
        cache.get_seq_length.return_value = 3
        with patch.object(inference, "DynamicCache", return_value=cache):
            self.assertIs(inference._make_cache(object()), cache)
        cache.activate_past_recording.assert_called_once_with()
        inference._crop_to(cache, 3)
        cache.crop.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()
