"""Aligned compressed payloads using the full original-unicast bit budget."""
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from bsrt_communication_evaluation.channel import hard_qpsk, qpsk_symbols, soft_qpsk
from bsrt_communication_evaluation.token_ids import ids_to_bits
from .transport import OriginalChannels


def compressed_allocation(original_tokens, kept_tokens, receivers, max_block_tokens=254):
    if not 0 < kept_tokens <= original_tokens or receivers not in (2, 3, 5):
        raise ValueError("Invalid token counts or receiver count")
    original_bits = 15*original_tokens
    budget = receivers*(original_bits+original_bits%2)
    count = math.ceil(kept_tokens/max_block_tokens)
    size, extra = divmod(kept_tokens, count)
    sizes = [size+int(i < extra) for i in range(count)]
    information = [15*t for t in sizes]
    # Allocate QPSK symbols proportionally; the final block absorbs rounding.
    wire = [2*((budget*k//sum(information))//2) for k in information]
    wire[-1] += budget-sum(wire)
    blocks = []
    for k, n in zip(information, wire, strict=True):
        mother = min(n, 2*((5*k)//2))
        if not .2 <= k/mother <= .95:
            raise ValueError("Unsupported LDPC mother-code rate")
        blocks.append({"k": k, "mother_n": mother, "wire_n": n,
                       "repeated_bits": n-mother})
    return {"source_tokens": original_tokens, "kept_tokens": kept_tokens,
            "actual_removal_percent": 100*(1-kept_tokens/original_tokens),
            "information_bits": sum(information), "budget_bits": budget,
            "transmitted_bits": sum(wire), "effective_code_rate": sum(information)/budget,
            "repeated_bits": sum(b["repeated_bits"] for b in blocks), "blocks": blocks}


@dataclass(frozen=True)
class CompressedFrame:
    information: np.ndarray
    encoded: np.ndarray
    symbols: np.ndarray
    fingerprint: str
    blocks: tuple
    coding: str = "ldpc"

    @property
    def payload_bits(self):
        return len(self.information)

    @property
    def transmitted_bits(self):
        return len(self.encoded)


class CompressedChannels(OriginalChannels):
    def make_compressed_frame(self, ids, original_tokens, receivers):
        plan = compressed_allocation(original_tokens, len(ids), receivers, self.config["max_block_tokens"])
        information = ids_to_bits(ids)
        blocks, encoded, offset = [], [], 0
        for b in plan["blocks"]:
            k, n, wire = b["k"], b["mother_n"], b["wire_n"]
            mother = self.encode_budget(information[offset:offset+k], n)
            encoded.append(np.resize(mother, wire))
            blocks.append((k, n, wire))
            offset += k
        encoded = np.concatenate(encoded)
        symbols = qpsk_symbols(encoded)
        for a in (information, encoded, symbols):
            a.setflags(write=False)
        return CompressedFrame(information, encoded, symbols,
                               hashlib.sha256(symbols.tobytes()).hexdigest(), tuple(blocks))

    def receive(self, frame, channel, snr, seed):
        y, variance, gain = self.physical(frame.symbols, channel, snr, seed)
        raw = hard_qpsk(y)[:len(frame.encoded)]
        llr = soft_qpsk(y, variance, gain)[:len(frame.encoded)]
        source_offset = coded_offset = failed = 0
        outputs = []
        for k, n, wire in frame.blocks:
            # Repeated observations refer to the same coded bit, not extra source tokens.
            combined = np.zeros(n, dtype=np.float64)
            np.add.at(combined, np.arange(wire)%n, llr[coded_offset:coded_offset+wire])
            _, decoder = self.codec(k, n)
            with self.torch.inference_mode():
                decoded = decoder(self.torch.tensor(combined[None], dtype=self.torch.float32))
            restored = decoded.cpu().numpy().astype(np.uint8).reshape(-1)
            outputs.append(restored)
            failed += int(np.any(restored != frame.information[source_offset:source_offset+k]))
            source_offset += k
            coded_offset += wire
        decoded = np.concatenate(outputs)
        if source_offset != len(frame.information) or coded_offset != len(frame.encoded):
            raise AssertionError("Incomplete source or wire framing")
        if hashlib.sha256(frame.symbols.tobytes()).hexdigest() != frame.fingerprint:
            raise AssertionError("Shared waveform changed")
        errors = int(np.count_nonzero(decoded != frame.information))
        return decoded, {"information_errors": errors, "information_ber": errors/len(decoded),
            "raw_coded_ber": float(np.mean(raw != frame.encoded)), "frame_error": int(errors > 0),
            "decode_success": int(errors == 0), "ldpc_blocks": len(frame.blocks),
            "ldpc_block_error": failed/len(frame.blocks), "ldpc_failed_blocks": failed}
