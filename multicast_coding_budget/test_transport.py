import unittest
import numpy as np

from bsrt_neural_compression.common import read_json, resolve
from .transport import OriginalChannels, allocation


class BudgetTests(unittest.TestCase):
    def test_budget_and_rate_limits(self):
        for tokens in [1, 35, 91, 254, 255, 256, 257, 418, 600]:
            for receivers in [2, 3, 5]:
                p = allocation(tokens, receivers)
                self.assertEqual(sum(b["tokens"] for b in p["blocks"]), tokens)
                self.assertEqual(p["information_bits"], 15*tokens)
                self.assertEqual(p["unicast_budget_bits"], receivers*(15*tokens+(15*tokens)%2))
                self.assertLessEqual(p["multicast_coded_bits"], p["unicast_budget_bits"])
                for b in p["blocks"]:
                    self.assertLessEqual(b["k"], 3810)
                    self.assertEqual(b["n"]%2, 0)
                    self.assertGreaterEqual(b["k"]/b["n"], .2)
                if receivers < 5:
                    self.assertEqual(p["unused_budget_bits"], 0)

    def test_no_semantic_token_removal(self):
        p = allocation(418, 5)
        self.assertEqual(p["source_tokens"], 418)
        self.assertEqual([b["tokens"] for b in p["blocks"]], [209, 209])


class SionnaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_json(resolve("multicast_coding_budget/config.json"))
        cls.channels = OriginalChannels(cls.config)

    def test_zero_pruning_round_trip(self):
        frame = self.channels.make_original_frame([1010, 2045]*40, 5, "ldpc")
        self.assertEqual(frame.blocks, ((1200, 6000),))
        encoder, decoder = self.channels.codec(1200, 6000)
        self.assertEqual(decoder.num_vns, encoder.n_ldpc)
        for channel in ["awgn", "fading"]:
            restored, stats = self.channels.receive(frame, channel, 60, 42)
            np.testing.assert_array_equal(restored, frame.information)
            self.assertEqual(stats["information_errors"], 0)

    def test_long_original_round_trip_all_group_sizes(self):
        ids = [1010, 2045, 3022]*139+[1010]
        self.assertEqual(len(ids), 418)
        for k in [2, 3, 5]:
            frame = self.channels.make_original_frame(ids, k, "ldpc")
            self.assertEqual(len(frame.blocks), 2)
            self.assertEqual(frame.payload_bits, 418*15)
            self.assertLessEqual(frame.transmitted_bits, allocation(418, k)["unicast_budget_bits"])
            self.assertFalse(frame.symbols.flags.writeable)
            for channel in ["awgn", "fading"]:
                restored, stats = self.channels.receive(frame, channel, 60, 42)
                np.testing.assert_array_equal(restored, frame.information)
                self.assertEqual(stats["ldpc_failed_blocks"], 0)
                self.assertEqual(stats["decode_success"], 1)

    def test_uncoded_same_original_information_and_padding(self):
        ids = [1010]*91
        unicast = self.channels.make_original_frame(ids, 5, "uncoded")
        multicast = self.channels.make_original_frame(ids, 5, "ldpc")
        np.testing.assert_array_equal(unicast.information, multicast.information)
        self.assertEqual(unicast.transmitted_bits, 1366)
        self.assertLessEqual(multicast.transmitted_bits, 5*unicast.transmitted_bits)
        decoded, stats = self.channels.receive(unicast, "awgn", 60, 42)
        np.testing.assert_array_equal(decoded, unicast.information)
        self.assertEqual(stats["information_errors"], 0)


if __name__ == "__main__":
    unittest.main()
