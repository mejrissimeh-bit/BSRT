"""Whole-context rate-matched LDPC with an exact original-QPSK bit budget."""
from collections import OrderedDict

import numpy as np

from experiments.communication.base.channel import Channels, hard_qpsk, qpsk_symbols, soft_qpsk


def budget_spec(original_id_count, compressed_id_count, width=15):
    if not 0 < compressed_id_count < original_id_count or width != 15:
        raise ValueError("Expected fewer compressed token IDs than original IDs, width=15")
    original_bits = int(original_id_count) * width
    k = int(compressed_id_count) * width
    n = original_bits + original_bits % 2
    rate = k / n
    if not 12 <= k <= 8448 or n > 316*384 or not .2 <= rate <= .95:
        raise ValueError(f"Unsupported Sionna LDPC dimensions: k={k}, n={n}")
    return {"original_payload_bits": original_bits, "original_budget_bits": n,
            "payload_bits": k, "codeword_bits": n, "redundancy_bits": n-k,
            "code_rate": rate, "coding_overhead_fraction": (n-k)/k,
            "actual_token_reduction": 1-compressed_id_count/original_id_count,
            "original_qpsk_padding_bits": n-original_bits}


class BudgetChannels(Channels):
    def __init__(self, config):
        super().__init__(config)
        self.codecs = OrderedDict()

    def codec(self, k, n):
        key = (int(k), int(n))
        if key not in self.codecs:
            from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
            encoder = LDPC5GEncoder(key[0], key[1], device="cpu")
            decoder = LDPC5GDecoder(encoder, hard_out=True, return_infobits=True,
                                    num_iter=self.config["ldpc_iterations"], device="cpu")
            self.codecs[key] = (encoder, decoder)
            if len(self.codecs) > 8:
                self.codecs.popitem(last=False)
        self.codecs.move_to_end(key)
        return self.codecs[key]

    def encode_budget(self, bits, budget_bits):
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        if not len(bits) or budget_bits % 2 or budget_bits <= len(bits):
            raise ValueError("Expected a larger, even original QPSK budget")
        encoder, _ = self.codec(len(bits), budget_bits)
        with self.torch.inference_mode():
            encoded = encoder(self.torch.tensor(bits[None], dtype=self.torch.float32))
        result = encoded.cpu().numpy().astype(np.uint8).reshape(-1)
        if len(result) != budget_bits:
            raise AssertionError("LDPC output violates the original transmission budget")
        return result

    def transmit_budget(self, bits, encoded, channel, snr_db, seed):
        if channel != "awgn":
            raise ValueError("This focused experiment supports AWGN only")
        bits, encoded = np.asarray(bits, dtype=np.uint8), np.asarray(encoded, dtype=np.uint8)
        if len(encoded) % 2 or len(encoded) <= len(bits):
            raise ValueError("Invalid budgeted codeword")
        _, decoder = self.codec(len(bits), len(encoded))
        symbols = qpsk_symbols(encoded)
        received, variance, gain = self.physical(symbols, channel, snr_db, seed)
        raw = hard_qpsk(received)
        llr = soft_qpsk(received, variance, gain)
        with self.torch.inference_mode():
            decoded = decoder(self.torch.tensor(llr[None], dtype=self.torch.float32))
        restored = decoded.cpu().numpy().astype(np.uint8).reshape(-1)
        if len(restored) != len(bits):
            raise AssertionError("Decoded information length mismatch")
        stats = {"transmitted_bits": 2*len(symbols), "qpsk_symbols": len(symbols),
                 "raw_coded_bit_error_rate": float(np.mean(raw != encoded)),
                 "decoded_information_ber": float(np.mean(restored != bits)),
                 "ldpc_blocks": 1, "padding_information_bits": 0}
        if stats["transmitted_bits"] != len(encoded):
            raise AssertionError("Modulation unexpectedly exceeded the coding budget")
        return restored, stats
