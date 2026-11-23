"""Arbitrary cluster sizes; fixed aggregate resources and soft repetition combining."""
import hashlib
import math

import numpy as np

from bsrt_communication_evaluation.channel import qpsk_symbols
from bsrt_communication_evaluation.token_ids import ids_to_bits
from multicast_coding_budget.compressed_transport import CompressedChannels, CompressedFrame
from multicast_coding_budget.transport import OriginalChannels, OriginalFrame


def budget_for_context(tokens):
    original = 15*int(tokens)
    original += original%2
    # Work in QPSK symbols: do not exceed 60% even after modulation padding.
    per_receiver = 2*((3*original)//10)
    kept = per_receiver//15
    if kept < 1:
        raise ValueError("Budget cannot carry a token")
    return {"original_tokens": int(tokens), "original_frame_bits": original,
            "per_receiver_budget_bits": per_receiver, "kept_tokens": kept,
            "actual_token_removal_percent": 100*(1-kept/tokens)}


def cluster_allocation(tokens, members):
    if members < 1:
        raise ValueError("Cluster cannot be empty")
    b = budget_for_context(tokens)
    wire = members*b["per_receiver_budget_bits"]
    k = 15*b["kept_tokens"]
    blocks = []
    if members > 1:
        count = math.ceil(b["kept_tokens"]/254)
        base, extra = divmod(b["kept_tokens"], count)
        sizes = [15*(base+int(i < extra)) for i in range(count)]
        lengths = [2*((wire*ki//k)//2) for ki in sizes]
        lengths[-1] += wire-sum(lengths)
        for ki, ni in zip(sizes, lengths, strict=True):
            mother = min(ni, 2*((5*ki)//2))
            if not .2 <= ki/mother <= .95:
                raise ValueError("Unsupported LDPC mother-code rate")
            blocks.append((ki, mother, ni))
    return {**b, "members": int(members), "budget_bits": wire, "information_bits": k,
            "transmitted_bits": wire, "code_rate": k/wire,
            "blocks": blocks, "repeated_bits": sum(ni-m for _, m, ni in blocks),
            "coding": "uncoded" if members == 1 else "ldpc_plus_repetition" if any(n > m for _, m, n in blocks) else "ldpc"}


class Channels(CompressedChannels):
    def make_frame(self, ids, original_tokens, members=1):
        p = cluster_allocation(original_tokens, members)
        if len(ids) != p["kept_tokens"]:
            raise ValueError("Token selection does not match the 60% budget")
        information = ids_to_bits(ids)
        if members == 1:
            encoded = np.pad(information, (0, p["budget_bits"]-len(information)))
        else:
            output, offset = [], 0
            for k, n, wire in p["blocks"]:
                mother = self.encode_budget(information[offset:offset+k], n)
                output.append(np.resize(mother, wire))
                offset += k
            encoded = np.concatenate(output)
        symbols = qpsk_symbols(encoded)
        for a in (information, encoded, symbols):
            a.setflags(write=False)
        fingerprint = hashlib.sha256(symbols.tobytes()).hexdigest()
        if members == 1:
            return OriginalFrame(information, encoded, symbols, "uncoded", fingerprint, ())
        return CompressedFrame(information, encoded, symbols, fingerprint, tuple(p["blocks"]))

    def receive(self, frame, channel, snr, seed):
        if frame.coding == "uncoded":
            return OriginalChannels.receive(self, frame, channel, snr, seed)
        return super().receive(frame, channel, snr, seed)
