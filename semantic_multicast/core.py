"""Fixed-rate resources and a reusable, immutable common QPSK waveform."""
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from bsrt_communication_evaluation.channel import hard_qpsk, qpsk_symbols, soft_qpsk
from bsrt_communication_evaluation.token_ids import ids_to_bits
from bsrt_budgeted_ldpc.channel import BudgetChannels


def stable_seed(*parts):
    value = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big") % (2**31-1)


def dimensions(tokens, coding, rate):
    if tokens < 1 or coding not in {"uncoded", "ldpc"} or not 0.2 <= rate <= 0.95:
        raise ValueError("Unsupported payload, coding or LDPC rate")
    k = 15 * int(tokens)
    n = 2 * math.ceil((k / rate if coding == "ldpc" else k) / 2)
    if coding == "ldpc" and (k > 8448 or n > 316*384):
        raise ValueError("Payload exceeds the reused Sionna 5G LDPC limits")
    return k, n


def allocate_unicast(tokens, receivers, coding, rate, regime):
    if regime == "equal_per_representation":
        return [tokens] * receivers
    if regime != "equal_aggregate":
        raise ValueError(regime)
    ceiling = dimensions(tokens, coding, rate)[1]
    counts = [1] * receivers
    if sum(dimensions(t, coding, rate)[1] for t in counts) > ceiling:
        raise ValueError("Shared budget cannot carry one token per receiver")
    # Water-fill integer token allocations; any remainder is explicitly unspent.
    while True:
        changed = False
        for index in range(receivers):
            candidate = counts.copy()
            candidate[index] += 1
            if sum(dimensions(t, coding, rate)[1] for t in candidate) <= ceiling:
                counts = candidate
                changed = True
        if not changed:
            return counts


def select_ids(token_ids, importance, count):
    importance = np.asarray(importance, dtype=float)
    if importance.shape != (len(token_ids),) or not np.isfinite(importance).all():
        raise ValueError("Nonfinite or misaligned token importance")
    if not 1 <= count <= len(token_ids):
        raise ValueError("Invalid token count")
    selected = np.argsort(-importance, kind="stable")[:count]
    return np.asarray(token_ids, dtype=int)[np.sort(selected)].tolist()


@dataclass(frozen=True)
class Frame:
    information: np.ndarray
    encoded: np.ndarray
    symbols: np.ndarray
    coding: str
    fingerprint: str

    @property
    def payload_bits(self):
        return len(self.information)

    @property
    def transmitted_bits(self):
        return 2 * len(self.symbols)


class BroadcastChannel(BudgetChannels):
    """Reuse the existing Sionna physical channel and variable-size LDPC codec."""

    def make_frame(self, token_ids, coding, rate):
        bits = ids_to_bits(token_ids)
        _, n = dimensions(len(token_ids), coding, rate)
        if coding == "ldpc":
            encoded = self.encode_budget(bits, n)
        else:
            encoded = np.pad(bits, (0, n-len(bits)))
        symbols = qpsk_symbols(encoded)
        if len(encoded) != n or 2*len(symbols) != n:
            raise AssertionError("Coding/modulation resource mismatch")
        for array in (bits, encoded, symbols):
            array.setflags(write=False)
        return Frame(bits, encoded, symbols, coding, hashlib.sha256(symbols.tobytes()).hexdigest())

    def receive(self, frame, channel, snr_db, seed):
        # No encoding or modulation occurs here. Each receiver observes the SAME frame.
        y, variance, gain = self.physical(frame.symbols, channel, snr_db, seed)
        raw = hard_qpsk(y)[:len(frame.encoded)]
        if frame.coding == "ldpc":
            _, decoder = self.codec(len(frame.information), len(frame.encoded))
            llr = soft_qpsk(y, variance, gain)[:len(frame.encoded)]
            with self.torch.inference_mode():
                restored = decoder(self.torch.tensor(llr[None], dtype=self.torch.float32))
            decoded = restored.cpu().numpy().astype(np.uint8).reshape(-1)
        else:
            decoded = raw[:len(frame.information)]
        if len(decoded) != len(frame.information):
            raise AssertionError("Information framing mismatch")
        if hashlib.sha256(frame.symbols.tobytes()).hexdigest() != frame.fingerprint:
            raise AssertionError("A receiver modified the common waveform")
        errors = int(np.count_nonzero(decoded != frame.information))
        return decoded, {
            "information_errors": errors,
            "information_ber": errors/len(decoded),
            "raw_coded_ber": float(np.mean(raw != frame.encoded)),
            "frame_error": int(errors > 0),
            "ldpc_block_error": int(errors > 0) if frame.coding == "ldpc" else None,
            "decode_success": int(errors == 0),
        }
