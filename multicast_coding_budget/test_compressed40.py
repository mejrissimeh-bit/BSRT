import unittest
import numpy as np

from bsrt_neural_compression.common import read_json, resolve
from .compressed_transport import CompressedChannels, compressed_allocation


class CompressedTests(unittest.TestCase):
    def test_exact_original_budget(self):
        for original in [35, 80, 91, 100, 160, 256, 285, 418, 600]:
            kept = original-int(original*.4)
            for k in [2, 3, 5]:
                p = compressed_allocation(original, kept, k)
                self.assertEqual(p["transmitted_bits"], k*(15*original+(15*original)%2))
                self.assertEqual(sum(b["k"] for b in p["blocks"]), 15*kept)
                self.assertEqual(p["repeated_bits"], sum(b["wire_n"]-b["mother_n"] for b in p["blocks"]))
                for b in p["blocks"]:
                    self.assertGreaterEqual(b["k"]/b["mother_n"], .2)
                    self.assertGreaterEqual(b["wire_n"], b["mother_n"])
                if k == 5:
                    self.assertGreater(p["repeated_bits"], 0)

    def test_round_trip_and_repetition(self):
        import torch
        torch.set_num_threads(2)
        channels = CompressedChannels(read_json(resolve("multicast_coding_budget/config.json")))
        for original in [100, 418]:
            kept = original-int(original*.4)
            for k in [2, 3, 5]:
                frame = channels.make_compressed_frame([1010, 2045]*(kept//2)+[1010]*(kept%2), original, k)
                self.assertEqual(frame.transmitted_bits, k*(15*original+(15*original)%2))
                offset = 0
                for _, mother, wire in frame.blocks:
                    codeword = frame.encoded[offset:offset+wire]
                    np.testing.assert_array_equal(codeword, np.resize(codeword[:mother], wire))
                    offset += wire
                for channel in ["awgn", "fading"]:
                    decoded, stats = channels.receive(frame, channel, 60, 42)
                    np.testing.assert_array_equal(decoded, frame.information)
                    self.assertEqual(stats["information_errors"], 0)


if __name__ == "__main__":
    unittest.main()
