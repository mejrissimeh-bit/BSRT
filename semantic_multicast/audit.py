"""Independently verify committed records and exact symbol/bit accounting."""
import argparse
import json

import numpy as np
import pandas as pd

from bsrt_neural_compression.common import read_json, resolve, write_json
from .core import allocate_unicast, dimensions, stable_seed


def audit(output):
    output = resolve(output)
    lock, plan = read_json(output/"run_lock.json"), read_json(output/"plan.json")
    config = lock["config"]
    groups = pd.read_csv(output/"group_results.csv")
    receivers = pd.read_csv(output/"receiver_results.csv")
    if len(groups) != plan["expected_cases"] or len(receivers) != plan["expected_receiver_outcomes"]:
        raise AssertionError("Evaluation is incomplete")
    if groups.case_id.duplicated().any() or receivers.duplicated(["case_id", "receiver_id"]).any():
        raise AssertionError("Duplicate saved rows")
    if set(groups.case_id) != set(receivers.case_id):
        raise AssertionError("Receiver/group case mismatch")
    grouped = {key: frame for key, frame in receivers.groupby("case_id")}
    manifests = {g["group_id"]: g for g in lock["groups"]}
    for group in groups.itertuples(index=False):
        part = grouped[group.case_id].sort_values("receiver_position")
        if len(part) != group.receiver_count or part.physical_seed.nunique() != group.receiver_count:
            raise AssertionError("Receiver count or independent channel seed mismatch")
        if part.receiver_id.astype(str).tolist() != list(map(str, manifests[group.group_id]["receiver_ids"])):
            raise AssertionError("Receiver identities changed")
        counts = json.loads(group.unicast_kept_tokens)
        expected_counts = allocate_unicast(group.multicast_kept_tokens, group.receiver_count,
                                           group.coding, config["ldpc_rate"], group.budget_regime)
        if counts != expected_counts:
            raise AssertionError("Unicast budget allocation differs from configuration")
        multicast = group.scenario == "multicast"
        expected = dimensions(group.multicast_kept_tokens, group.coding, config["ldpc_rate"])[1]
        if group.aggregate_ceiling_bits != expected:
            raise AssertionError("Aggregate comparison ceiling is wrong")
        for i, receiver in enumerate(part.itertuples(index=False)):
            tokens = group.multicast_kept_tokens if multicast else counts[i]
            k, n = dimensions(tokens, group.coding, config["ldpc_rate"])
            if receiver.payload_bits != k or receiver.transmitted_bits != n or receiver.qpsk_symbols*2 != n:
                raise AssertionError("Saved encoded/modulated resource calculation failed")
            if receiver.physical_seed != stable_seed(group.context_id, str(receiver.receiver_id), group.channel,
                                                      float(group.snr_db), group.snr_profile, group.channel_seed):
                raise AssertionError("Saved receiver seed is wrong")
            expected_snr = group.snr_db
            if group.snr_profile == "heterogeneous":
                expected_snr += np.linspace(-config["heterogeneous_span_db"]/2,
                                            config["heterogeneous_span_db"]/2, group.receiver_count)[i]
            if not np.isclose(receiver.receiver_snr_db, expected_snr):
                raise AssertionError("Saved heterogeneous SNR is wrong")
            if receiver.decode_success != int(receiver.information_errors == 0):
                raise AssertionError("Decoding success is not bit-perfect equality")
            if not np.isclose(receiver.information_ber, receiver.information_errors/k):
                raise AssertionError("BER denominator is wrong")
        if multicast:
            if part.waveform_sha256.nunique() != 1 or group.transmission_count != 1:
                raise AssertionError("Not a common-waveform multicast transmission")
            bit_total, info_total = part.transmitted_bits.iloc[0], part.payload_bits.iloc[0]
        else:
            if group.transmission_count != group.receiver_count:
                raise AssertionError("Unicast transmissions were not counted independently")
            bit_total, info_total = part.transmitted_bits.sum(), part.payload_bits.sum()
        if group.total_transmitted_bits != bit_total or group.total_qpsk_symbols*2 != bit_total:
            raise AssertionError("Aggregate transmission resources were over/undercounted")
        if group.aggregate_information_bits != info_total:
            raise AssertionError("Aggregate information payload was over/undercounted")
        if group.budget_regime == "equal_aggregate" and group.total_transmitted_bits > expected:
            raise AssertionError("Aggregate constraint violated")
        if group.all_receivers_decode != int(part.decode_success.all()):
            raise AssertionError("All-receiver success is not an observed conjunction")
        if not np.isclose(group.mean_accuracy, part.accuracy.mean()):
            raise AssertionError("Group semantic means disagree")
    pairs = pd.read_csv(output/"paired_comparison.csv")
    if len(pairs)*2 != len(groups):
        raise AssertionError("Unicast/multicast pairing incomplete")
    same = pairs[pairs.budget_regime.eq("equal_per_representation")]
    if not np.allclose(same.resource_savings, 1-1/same.receiver_count):
        raise AssertionError("Equal message-size savings do not match actual symbols")
    equal = pairs[pairs.budget_regime.eq("equal_aggregate")]
    if not (equal.total_qpsk_symbols_unicast <= equal.total_qpsk_symbols_multicast).all():
        raise AssertionError("Paired aggregate budget violated")
    result = {"passed": True, "cases_checked": len(groups), "receiver_outcomes_checked": len(receivers),
              "paired_conditions_checked": len(pairs), "contexts": plan["contexts"],
              "checks": ["complete paired cohort", "distinct receiver channel seeds", "shared waveform hashes",
                         "fixed-width payload", "actual coded/QPSK dimensions", "aggregate accounting",
                         "equal-aggregate ceiling", "BER denominator", "bit-perfect success", "observed all-receiver success"]}
    write_json(output/"verification.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="semantic_multicast/results/pilot")
    args = parser.parse_args()
    audit(args.output)
