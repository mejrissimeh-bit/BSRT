"""Fixed-width tokenizer IDs, with ideal externally known payload framing."""
import numpy as np


def ids_to_bits(ids, width=15):
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    if np.any(ids < 0) or np.any(ids >= 2**width):
        raise ValueError("Token ID does not fit the configured width")
    return ((ids[:, None] >> np.arange(width-1, -1, -1)) & 1).astype(np.uint8).reshape(-1)


def unpack_ids(bits, width=15):
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if len(bits) % width or np.any(bits > 1):
        raise ValueError("Expected a complete fixed-width binary token stream")
    return bits.reshape(-1, width).astype(np.int64) @ (1 << np.arange(width-1, -1, -1))


def token_erasure_mask(bit_count, probability, packet_size, seed, width=15):
    if bit_count % width:
        raise ValueError("Partial token in framed source")
    lost = np.random.default_rng(seed).random(int(np.ceil(bit_count/packet_size))) < probability
    return np.repeat(lost, packet_size)[:bit_count].reshape(-1, width).any(axis=1)


def received_ids(bits, tokenizer, width=15, erased_tokens=None):
    raw = unpack_ids(bits, width)
    invalid = raw >= len(tokenizer)
    reserved = set(tokenizer.all_special_ids) - {tokenizer.unk_token_id}
    invalid |= np.isin(raw, list(reserved))
    known_erased = np.zeros(len(raw), dtype=bool) if erased_tokens is None else np.asarray(erased_tokens, dtype=bool)
    if known_erased.shape != raw.shape:
        raise ValueError("Erasure mask does not match received tokens")
    usable = raw.copy()
    usable[invalid | known_erased] = tokenizer.unk_token_id
    return raw, usable.tolist(), {"invalid_or_reserved_token_fraction": float(invalid.mean()) if len(raw) else 0.,
                                "known_erased_token_fraction": float(known_erased.mean()) if len(raw) else 0.}


def decode_ids(ids, tokenizer):
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
