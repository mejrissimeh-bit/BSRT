from __future__ import annotations

import inspect
import unittest

import numpy as np
import torch

from .model import BSRT, RetentionNetwork, budget_mask, lexical_groups, render_context


class FeaturesStub:
    def get(self, context):
        n = len(context.split())
        return {"embeddings": torch.arange(n*8, dtype=torch.float32).reshape(n, 8)/100,
                "structure": torch.zeros(n, 8), "groups": [[i] for i in range(n)],
                "offsets": [(i*2, i*2+1) for i in range(n)]}


class BSRTTests(unittest.TestCase):
    def test_budget_and_wordpiece_integrity(self):
        rng = np.random.default_rng(5)
        groups = [[0], [1, 2], [3], [4, 5, 6], [7], [8], [9]]
        for removal in [.05, .1, .2, .3, .4, .5]:
            mask = budget_mask(rng.normal(size=10), groups, removal)
            self.assertEqual(int(mask.sum()), 10-int(10*removal))
            for group in groups:
                self.assertTrue(mask[group].all() or not mask[group].any())

    def test_selection_is_optimal_for_feasible_budget(self):
        from itertools import product
        groups, scores = [[0, 1], [2], [3], [4]], np.array([1., 2., 8., -3., 5.])
        mask = budget_mask(scores, groups, .4)
        feasible = []
        for bits in product([0, 1], repeat=len(groups)):
            ids = [i for keep, group in zip(bits, groups) if keep for i in group]
            if len(ids) == 3:
                feasible.append(scores[ids].sum())
        self.assertEqual(scores[mask].sum(), max(feasible))

    def test_context_mask_is_question_independent(self):
        torch.manual_seed(7)
        network = RetentionNetwork(input_dim=8, hidden_dim=12, heads=3, layers=1, dropout=.3)
        policy = BSRT(network, FeaturesStub())
        context = "a b c d e f g h i j"
        masks = []
        for question in ["Who?", "When?", "Where?"]:
            # Questions are deliberately held outside the compressor call.
            masks.append(policy.select(context, .4)[0])
        self.assertTrue(all(np.array_equal(masks[0], mask) for mask in masks))
        self.assertEqual(list(inspect.signature(policy.compress).parameters), ["context", "removal_ratio"])
        with self.assertRaises(TypeError):
            policy.compress(context, .4, question="Who?")

    def test_order_and_spelling(self):
        context = "Washington founded a school."
        offsets = [(0, 10), (11, 18), (19, 20), (21, 27)]
        text = render_context(context, offsets, [True, True, False, True])
        self.assertEqual(text, "Washington founded school")

    def test_invalid_budget(self):
        for ratio in [-.1, 1.0]:
            with self.assertRaises(ValueError):
                budget_mask(np.array([1, 2]), [[0], [1]], ratio)

    def test_disabled_structure_and_budget_do_not_affect_scores(self):
        torch.manual_seed(7)
        network = RetentionNetwork(input_dim=8, hidden_dim=12, heads=3, layers=1,
                                   budget_conditioning=False, structural_features=False).eval()
        hidden = torch.randn(10, 8)
        first = network(hidden, torch.zeros(10, 8), .1)
        second = network(hidden, torch.ones(10, 8)*99, .5)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertFalse(any(p.requires_grad for p in network.budget.parameters()))
        self.assertFalse(any(p.requires_grad for p in network.structure.parameters()))

    def test_reward_only_training_never_reads_answer_spans(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from .training import fit_stage
        network = RetentionNetwork(input_dim=8, hidden_dim=12, heads=3, layers=1)
        record = {"context_id": "a", "context": "a b c d e f g h i j"}
        target = {("a", 40): [{"target": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], "weight": 1.0}]}
        with tempfile.TemporaryDirectory() as directory:
            config = {"output_dir": directory, "seed": 7, "learning_rate": .001,
                      "distill_epochs": 1, "pretrain_epochs": 0, "removal_percentages": [40],
                      "answer_span_supervision": False, "auxiliary_weight": 0, "budget_weight": .1}
            with patch("bsrt_neural_compression.training.span_targets", side_effect=AssertionError("Answer labels leaked")), patch("bsrt_neural_compression.training.validate", return_value=(.5, [])):
                fit_stage(network, FeaturesStub(), type("QA", (), {"calls": 0})(), config, [record], [], "distill_0", target)
            self.assertTrue((Path(directory)/"checkpoints/best.pt").exists())

    def test_input_ablation_removes_contextual_projection_including_bias(self):
        network = RetentionNetwork(input_dim=8, hidden_dim=12, heads=3, layers=1,
                                   contextual_vectors=False).eval()
        structure = torch.randn(10, 8)
        with torch.no_grad():
            first = network(torch.randn(10, 8), structure, .4)
            network.project.bias.fill_(999)
            second = network(torch.randn(10, 8)*99, structure, .4)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertFalse(any(p.requires_grad for p in network.project.parameters()))

    def test_individual_structural_feature_ablation(self):
        for index in [3, 4]:
            network = RetentionNetwork(input_dim=8, hidden_dim=12, heads=3, layers=1,
                                       disabled_structural_features=[index]).eval()
            hidden, structure = torch.randn(10, 8), torch.randn(10, 8)
            changed = structure.clone()
            changed[:, index] += 999
            first = network(hidden, structure, .4)
            second = network(hidden, changed, .4)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            first.sum().backward()
            self.assertEqual(network.structure.weight.grad[:, index].abs().sum().item(), 0)
            self.assertGreater(network.structure.weight.grad[:, (index+1) % 8].abs().sum().item(), 0)
            self.assertNotIn("structural_input_mask", network.state_dict())

    def test_invalid_structural_feature_indices(self):
        for invalid in [[-1], [8], [3, 3], [True]]:
            with self.assertRaises(ValueError):
                RetentionNetwork(disabled_structural_features=invalid)

    def test_input_ablation_removes_learned_position_embedding(self):
        network = RetentionNetwork(input_dim=8, hidden_dim=12, heads=3, layers=1,
                                   position_embedding=False).eval()
        hidden, structure = torch.randn(10, 8), torch.randn(10, 8)
        with torch.no_grad():
            first = network(hidden, structure, .4)
            network.positions.weight.fill_(999)
            second = network(hidden, structure, .4)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertFalse(any(p.requires_grad for p in network.positions.parameters()))

    def test_default_forward_preserves_reference_calculation(self):
        network = RetentionNetwork(input_dim=8, hidden_dim=12, heads=3, layers=1).eval()
        hidden, structure = torch.randn(10, 8), torch.randn(10, 8)
        with torch.no_grad():
            values = network.project(hidden) + network.structure(structure) + network.positions(torch.arange(10))
            values = values + network.budget(torch.tensor([.4, .6]))
            expected = network.out(network.transformer(values.unsqueeze(0))).squeeze(0).squeeze(-1)
            actual = network(hidden, structure, .4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_split_articles_and_final_contexts_are_disjoint(self):
        from .common import load_config, resolve
        from .data import split_manifest, final_records
        from .common import digest
        config = load_config(resolve("bsrt_neural_compression/config_pilot.json"))
        path = resolve(config["output_dir"]) / "splits.json"
        if not path.exists():
            self.skipTest("Run prepare to create the data manifest")
        manifest = split_manifest(config)
        final = {digest(row["context"]) for row in final_records(config)}
        articles = []
        contexts = []
        for name in ["train", "validation", "untouched"]:
            current = {row["context_id"] for row in manifest[name]}
            self.assertFalse(current & final)
            articles.append({row["article"] for row in manifest[name]})
            contexts.append(current)
        for first in range(3):
            for second in range(first):
                self.assertFalse(articles[first] & articles[second])
                self.assertFalse(contexts[first] & contexts[second])

    def test_trained_head_is_question_independent(self):
        from .common import load_config, resolve, digest, read_json
        config = load_config(resolve("bsrt_neural_compression/config_pilot.json"))
        checkpoint = resolve(config["output_dir"]) / "checkpoints" / "best.pt"
        if not checkpoint.exists():
            self.skipTest("A trained checkpoint is not yet available")
        record = read_json(resolve(config["output_dir"]) / "splits.json")["validation"][0]
        saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
        network = RetentionNetwork(**config["model"])
        network.load_state_dict(saved["state_dict"])
        candidates = list(resolve(config["cache_dir"]).glob("embeddings_*/"+digest(record["context"])+".pt"))
        self.assertTrue(candidates)

        class CachedFeatures:
            def get(self, context):
                self_context = digest(context)
                assert self_context == record["context_id"]
                return torch.load(candidates[0], weights_only=False, map_location="cpu")

        policy = BSRT(network, CachedFeatures())
        results = [policy.compress(record["context"], .4) for _ in record["questions"]]
        self.assertEqual(len(set(results)), 1)


if __name__ == "__main__":
    unittest.main()
