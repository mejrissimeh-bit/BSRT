import unittest
import numpy as np

from .core import BroadcastChannel, allocate_unicast, dimensions, select_ids, stable_seed
from .run import evaluate_case


class ResourceTests(unittest.TestCase):
    def test_qpsk_padding(self):
        self.assertEqual(dimensions(3, "uncoded", .5), (45, 46))
        self.assertEqual(dimensions(3, "ldpc", .5), (45, 90))

    def test_equal_representation(self):
        for k in [2, 3, 5]:
            counts = allocate_unicast(153, k, "ldpc", .5, "equal_per_representation")
            total = sum(dimensions(t, "ldpc", .5)[1] for t in counts)
            self.assertAlmostEqual(1-dimensions(153, "ldpc", .5)[1]/total, 1-1/k)

    def test_aggregate_accounting(self):
        for coding in ["uncoded", "ldpc"]:
            for k in [2, 3, 5]:
                for tokens in [7, 91, 153]:
                    counts = allocate_unicast(tokens, k, coding, .7, "equal_aggregate")
                    budget = dimensions(tokens, coding, .7)[1]
                    used = sum(dimensions(t, coding, .7)[1] for t in counts)
                    self.assertLessEqual(used, budget)
                    for index in range(k):
                        more = counts.copy()
                        more[index] += 1
                        self.assertGreater(sum(dimensions(t, coding, .7)[1] for t in more), budget)

    def test_selection_order_and_ties(self):
        self.assertEqual(select_ids([10, 11, 12, 13], [1, 5, 5, 0], 2), [11, 12])
        self.assertEqual(select_ids([10, 11, 12, 13], [1, 5, 5, 0], 3), [10, 11, 12])

    def test_receiver_seeds_paired_but_distinct(self):
        self.assertEqual(stable_seed("c", "q", "awgn", 3., "homogeneous", 42),
                         stable_seed("c", "q", "awgn", 3., "homogeneous", 42))
        self.assertNotEqual(stable_seed("c", "q1", 42), stable_seed("c", "q2", 42))


class PhysicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bsrt_neural_compression.common import read_json, resolve
        cls.config = read_json(resolve("semantic_multicast/config.json"))
        cls.channel = BroadcastChannel(cls.config)

    def test_sionna_noiseless_limit(self):
        for coding in ["uncoded", "ldpc"]:
            frame = self.channel.make_frame([1010, 2045, 3022]*10, coding, .5)
            for channel in ["awgn", "fading"]:
                decoded, stats = self.channel.receive(frame, channel, 60, 42)
                np.testing.assert_array_equal(decoded, frame.information)
                self.assertEqual(stats["decode_success"], 1)
                self.assertEqual(frame.transmitted_bits, len(frame.encoded))
                self.assertFalse(frame.symbols.flags.writeable)

    def test_shared_frame_and_independent_receivers(self):
        frame = self.channel.make_frame([1010, 2045, 3022]*10, "uncoded", .5)
        group = {"context_id": "example", "k": 2}
        questions = [{"id": str(i), "question": str(i), "answers": {"text": ["x"]}} for i in range(2)]
        seen = []

        def transport(actual, channel, snr, seed):
            seen.append((id(actual), seed))
            decoded, stats = self.channel.receive(actual, channel, snr, seed)
            return decoded, stats, "x"

        def metric(text, question):
            return {"accuracy": 1, "f1": 1., "semantic_similarity": 1., "prediction": "x"}

        aggregate, receivers = evaluate_case(self.config, group, [frame], questions, "multicast",
                                               "awgn", -5, "heterogeneous", 42, transport, metric)
        self.assertEqual(seen[0][0], seen[1][0])
        self.assertNotEqual(seen[0][1], seen[1][1])
        self.assertEqual(aggregate["transmission_count"], 1)
        self.assertEqual(aggregate["total_qpsk_symbols"], len(frame.symbols))
        self.assertEqual(aggregate["all_receivers_decode"], 0)
        self.assertEqual(len(receivers), 2)
        self.assertEqual(receivers[1]["receiver_snr_db"]-receivers[0]["receiver_snr_db"], 6)

    def test_seed_reproducibility_and_distinct_noise(self):
        frame = self.channel.make_frame([1010, 2045, 3022]*10, "uncoded", .5)
        for channel in ["awgn", "fading"]:
            a, _ = self.channel.receive(frame, channel, 0, 123)
            b, _ = self.channel.receive(frame, channel, 0, 123)
            c, _ = self.channel.receive(frame, channel, 0, 2024)
            np.testing.assert_array_equal(a, b)
            self.assertFalse(np.array_equal(a, c))

    def test_awgn_ber_scale(self):
        frame = self.channel.make_frame([1010]*2000, "uncoded", .5)
        _, stats = self.channel.receive(frame, "awgn", 0, 42)
        # Gray QPSK at Es/N0=1 has bit error probability Q(1)=0.158655.
        self.assertAlmostEqual(stats["information_ber"], .158655, delta=.015)


if __name__ == "__main__":
    unittest.main()
