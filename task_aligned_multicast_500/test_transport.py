import unittest
import numpy as np

from bsrt_neural_compression.common import read_json, resolve
from .transport import Channels, budget_for_context, cluster_allocation


class BudgetTests(unittest.TestCase):
    def test_sixty_percent_budget(self):
        for tokens in [35, 91, 160, 256, 285, 299, 418]:
            b = budget_for_context(tokens)
            self.assertLessEqual(b["per_receiver_budget_bits"], .6*b["original_frame_bits"])
            self.assertLessEqual(15*b["kept_tokens"], b["per_receiver_budget_bits"])
            for members in range(1, 30):
                p = cluster_allocation(tokens, members)
                self.assertEqual(p["transmitted_bits"], members*b["per_receiver_budget_bits"])
                self.assertEqual(p["information_bits"], 15*b["kept_tokens"])
                if members > 1:
                    self.assertEqual(sum(w for _, _, w in p["blocks"]), p["budget_bits"])
                    for k, mother, wire in p["blocks"]:
                        self.assertGreaterEqual(k/mother, .2)
                        self.assertGreaterEqual(wire, mother)

    def test_full_cluster_sum_equals_unicast(self):
        b = budget_for_context(418)["per_receiver_budget_bits"]
        for sizes in [[1, 1], [2, 3, 1], [12, 8, 4, 3, 1, 1]]:
            self.assertEqual(sum(cluster_allocation(418, n)["transmitted_bits"] for n in sizes), sum(sizes)*b)

    def test_noiseless_round_trip(self):
        import torch
        torch.set_num_threads(2)
        channels = Channels(read_json(resolve("task_aligned_multicast_500/config.json")))
        for tokens, members in [(91, 1), (91, 2), (91, 5), (91, 29), (418, 3)]:
            keep = budget_for_context(tokens)["kept_tokens"]
            ids = [1010, 2045]*(keep//2)+[1010]*(keep%2)
            frame = channels.make_frame(ids, tokens, members)
            for channel in ["awgn", "fading"]:
                decoded, stats = channels.receive(frame, channel, 60, 42)
                np.testing.assert_array_equal(decoded, frame.information)
                self.assertEqual(stats["decode_success"], 1)


if __name__ == "__main__":
    unittest.main()
