"""Spend the K-copy uncoded budget on segmented, shared Sionna LDPC frames."""
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from bsrt_communication_evaluation.channel import hard_qpsk, qpsk_symbols, soft_qpsk
from bsrt_communication_evaluation.token_ids import ids_to_bits
from bsrt_budgeted_ldpc.channel import BudgetChannels


def allocation(tokens, receivers, max_block_tokens=254):
    if tokens < 1 or receivers not in [2, 3, 5] or not 1 <= max_block_tokens <= 254:
        raise ValueError("Expected a nonempty payload, K=2/3/5, and BG2-safe blocks")
    k = 15*int(tokens)
    uncoded = k+k%2
    budget = int(receivers)*uncoded
    blocks = math.ceil(tokens/max_block_tokens)
    size, extra = divmod(tokens, blocks)
    sizes = [size+int(i < extra) for i in range(blocks)]
    information = [15*t for t in sizes]
    encoded = [2*((receivers*ki)//2) for ki in information]
    maximum = [2*((5*ki)//2) for ki in information]
    remaining = budget-sum(encoded)
    for i in range(blocks):
        add = min(remaining, maximum[i]-encoded[i])
        add -= add%2
        encoded[i] += add
        remaining -= add
    if sum(encoded) > budget or any(ni%2 or not .2 <= ki/ni <= .95 for ki, ni in zip(information, encoded, strict=True)):
        raise AssertionError("Invalid code rate or exceeded unicast budget")
    return {"source_tokens": int(tokens), "information_bits": k, "uncoded_frame_bits": uncoded,
            "unicast_budget_bits": budget, "multicast_coded_bits": sum(encoded),
            "unused_budget_bits": remaining, "effective_multicast_rate": k/sum(encoded),
            "blocks": [{"tokens": t, "k": ki, "n": ni} for t, ki, ni in zip(sizes, information, encoded, strict=True)]}


@dataclass(frozen=True)
class OriginalFrame:
    information: np.ndarray
    encoded: np.ndarray
    symbols: np.ndarray
    coding: str
    fingerprint: str
    blocks: tuple

    @property
    def payload_bits(self):
        return len(self.information)

    @property
    def transmitted_bits(self):
        return len(self.encoded)


class OriginalChannels(BudgetChannels):
    def codec(self, k, n):
        encoder, decoder = super().codec(k, n)
        # Sionna 2.0.1 slices the PCM with [:-0] when no nodes are pruned.
        if decoder.num_vns == 0:
            from sionna.phy.fec.ldpc import LDPC5GDecoder
            decoder = LDPC5GDecoder(encoder, hard_out=True, return_infobits=True,
                                    num_iter=self.config["ldpc_iterations"],
                                    prune_pcm=False, device="cpu")
            self.codecs[(int(k), int(n))] = (encoder, decoder)
        return encoder, decoder

    def make_original_frame(self, ids, receivers, coding):
        plan = allocation(len(ids), receivers, self.config["max_block_tokens"])
        information = ids_to_bits(ids)
        if coding == "uncoded":
            encoded = np.pad(information, (0, len(information)%2))
            blocks = ()
        elif coding == "ldpc":
            blocks = tuple((block["k"], block["n"]) for block in plan["blocks"])
            offset, outputs = 0, []
            for k, n in blocks:
                outputs.append(self.encode_budget(information[offset:offset+k], n))
                offset += k
            encoded = np.concatenate(outputs)
            if offset != len(information):
                raise AssertionError("LDPC blocks do not cover all original tokens")
        else:
            raise ValueError(coding)
        symbols = qpsk_symbols(encoded)
        if 2*len(symbols) != len(encoded):
            raise AssertionError("Unexpected modulation padding")
        for data in [information, encoded, symbols]:
            data.setflags(write=False)
        return OriginalFrame(information, encoded, symbols, coding,
                             hashlib.sha256(symbols.tobytes()).hexdigest(), blocks)

    def receive(self, frame, channel, snr, seed):
        y, variance, gain = self.physical(frame.symbols, channel, snr, seed)
        raw = hard_qpsk(y)[:len(frame.encoded)]
        failed_blocks = 0
        if frame.coding == "uncoded":
            decoded = raw[:len(frame.information)]
        else:
            llr = soft_qpsk(y, variance, gain)[:len(frame.encoded)]
            source_offset = coded_offset = 0
            outputs = []
            for k, n in frame.blocks:
                _, decoder = self.codec(k, n)
                with self.torch.inference_mode():
                    result = decoder(self.torch.tensor(llr[coded_offset:coded_offset+n][None], dtype=self.torch.float32))
                restored = result.cpu().numpy().astype(np.uint8).reshape(-1)
                failed_blocks += int(np.any(restored != frame.information[source_offset:source_offset+k]))
                outputs.append(restored)
                source_offset += k
                coded_offset += n
            decoded = np.concatenate(outputs)
        if len(decoded) != len(frame.information):
            raise AssertionError("Decoded source length mismatch")
        if hashlib.sha256(frame.symbols.tobytes()).hexdigest() != frame.fingerprint:
            raise AssertionError("A receiver changed the shared waveform")
        errors = int(np.count_nonzero(decoded != frame.information))
        return decoded, {"information_errors": errors, "information_ber": errors/len(decoded),
            "raw_coded_ber": float(np.mean(raw != frame.encoded)), "frame_error": int(errors > 0),
            "decode_success": int(errors == 0), "ldpc_blocks": len(frame.blocks),
            "ldpc_block_error": failed_blocks/len(frame.blocks) if frame.blocks else None,
            "ldpc_failed_blocks": failed_blocks}
