import numpy as np

from experiments.communication.base.channel import hard_qpsk, qpsk_symbols
from experiments.communication.budgeted_ldpc.channel import budget_spec


def test_qpsk_round_trip_without_noise():
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
    recovered = hard_qpsk(qpsk_symbols(bits))
    assert np.array_equal(recovered, bits)


def test_compression_savings_become_ldpc_redundancy():
    spec = budget_spec(original_id_count=100, compressed_id_count=70)
    assert spec["original_budget_bits"] == 1500
    assert spec["payload_bits"] == 1050
    assert spec["redundancy_bits"] == 450
    assert spec["code_rate"] == 0.7
