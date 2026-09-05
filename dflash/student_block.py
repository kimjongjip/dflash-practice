"""Student exercise: parallel block drafting with the pretrained DFlash model."""


def parallel_block_draft(
    model,
    target_hidden,
    noise_embedding,
    draft_position_ids,
    past_key_values_draft,
    output_head,
    verify_size,
):
    """Return torch.long tokens of shape [1, verify_size - 1] on the input device.

    Use one drafter forward over the prepared slots, select the final
    verify_size - 1 hidden states, and apply the target LM head via
    model.compute_logits before selecting all positions greedily together.
    The caller prepares inputs and manages both KV caches and verification.
    This hook is called only for original DFlash greedy blocks of size > 1.
    Replace this module attribute from your notebook; no reload is needed.
    """
    raise NotImplementedError(
        "Implement parallel_block_draft in your notebook and assign it to "
        "dflash.student_block.parallel_block_draft before greedy generation."
    )
