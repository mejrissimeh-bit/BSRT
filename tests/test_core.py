from pathlib import Path
import unittest

import numpy as np
import torch
from safetensors.torch import load_file

from bsrt_neural_compression import RetentionNetwork, budget_mask, render_context


ROOT = Path(__file__).resolve().parents[1]


def test_budget_mask_finds_optimal_whole_group_subset():
    scores = np.array([6.0, 6.0, 7.0, 3.0])
    groups = [[0, 1], [2], [3]]
    mask = budget_mask(scores, groups, removal_ratio=0.5)
    assert mask.tolist() == [True, True, False, False]
    assert mask[0] == mask[1]


def test_render_context_preserves_order_and_original_spelling():
    context = "Vision transformers preserve structure."
    offsets = [(0, 6), (7, 19), (20, 28), (29, 38)]
    mask = [True, True, False, True]
    assert render_context(context, offsets, mask) == "Vision transformers structure"


def test_invalid_budget_is_rejected():
    for ratio in (-0.1, 1.0, float("nan")):
        with unittest.TestCase().assertRaises(ValueError):
            budget_mask([1.0], [[0]], ratio)


def test_released_weights_load_and_run_deterministically():
    config = {
        "input_dim": 768,
        "hidden_dim": 96,
        "layers": 2,
        "heads": 4,
        "dropout": 0.1,
        "contextual": True,
        "budget_conditioning": True,
        "structural_features": True,
        "contextual_vectors": True,
        "position_embedding": True,
    }
    network = RetentionNetwork(**config)
    state = load_file(ROOT / "checkpoints" / "bsrt_a0.safetensors")
    network.load_state_dict(state)
    network.eval()
    torch.manual_seed(7)
    embeddings = torch.randn(9, 768)
    structure = torch.randn(9, 8)
    with torch.inference_mode():
        first = network(embeddings, structure, 0.4)
        second = network(embeddings, structure, 0.4)
    assert first.shape == (9,)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
