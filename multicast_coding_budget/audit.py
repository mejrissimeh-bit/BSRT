"""Check saved original-payload resource accounting and receiver-side diagnostics."""
import argparse
import json

import numpy as np
import pandas as pd

from bsrt_neural_compression.common import read_json, resolve, write_json
from semantic_multicast.core import stable_seed
from .transport import allocation


def audit(output):
    output = resolve(output)
    config = read_json(output/"run_lock.json")["config"]
    plan = read_json(output/"plan.json")
    groups = pd.read_csv(output/"group_results.csv")
    receivers = pd.read_csv(output/"receiver_results.csv")
    if len(groups) != plan["expected_cases"] or len(receivers) != plan["expected_receiver_outcomes"]:
        raise ValueError("Incomplete evaluation")
    if groups.case_id.duplicated().any() or receivers.duplicated(["case_id", "receiver_id"]).any():
        raise ValueError("Duplicate outcomes")
    by_case = {k: v.sort_values("receiver_position") for k, v in receivers.groupby("case_id")}
    for g in groups.itertuples(index=False):
        p = allocation(g.source_token_count, g.receiver_count, config["max_block_tokens"])
        r = by_case[g.case_id]
        if len(r) != g.receiver_count or r.physical_seed.nunique() != g.receiver_count:
            raise AssertionError("Receiver count/independence mismatch")
        if not r.payload_bits.eq(p["information_bits"]).all() or g.removal_percent != 0:
            raise AssertionError("A scenario removed original source tokens")
        unicast = g.scenario == "unicast"
        n = p["uncoded_frame_bits"] if unicast else p["multicast_coded_bits"]
        copies = g.receiver_count if unicast else 1
        if g.transmission_count != copies or g.total_transmitted_bits != copies*n:
            raise AssertionError("Wrong unicast/multicast aggregate accounting")
        if not r.transmitted_bits.eq(n).all() or not r.qpsk_symbols.eq(n//2).all():
            raise AssertionError("Saved receiver waveform length is wrong")
        if g.total_qpsk_symbols*2 != g.total_transmitted_bits or g.budget_bits != p["unicast_budget_bits"]:
            raise AssertionError("Aggregate symbols/budget mismatch")
        if g.total_transmitted_bits > g.budget_bits or g.unused_budget_bits != g.budget_bits-g.total_transmitted_bits:
            raise AssertionError("Budget exceeded or remainder hidden")
        if r.waveform_sha256.nunique() != 1:
            raise AssertionError("Original shared waveform identity changed")
        if g.all_receivers_decode != int(r.decode_success.all()):
            raise AssertionError("All-receiver success is not the observed conjunction")
        for i, row in enumerate(r.itertuples(index=False)):
            snr = g.snr_db + np.linspace(-config["heterogeneous_span_db"]/2,
                                         config["heterogeneous_span_db"]/2, g.receiver_count)[i]
            if not np.isclose(row.receiver_snr_db, snr):
                raise AssertionError("Incorrect heterogeneous receiver SNR")
            if row.physical_seed != stable_seed(g.context_id, str(row.receiver_id), g.channel,
                                               float(g.snr_db), "heterogeneous", g.channel_seed):
                raise AssertionError("Receiver seed was not paired across scenarios")
            if not np.isclose(row.information_ber, row.information_errors/row.payload_bits):
                raise AssertionError("BER denominator mismatch")
            if row.decode_success != int(row.information_errors == 0):
                raise AssertionError("Decode success is not bit-perfect equality")
            if row.ldpc_blocks != (0 if unicast else len(p["blocks"])):
                raise AssertionError("Missing/extra LDPC block")
    match = ["context_id", "group_id", "receiver_count", "channel", "snr_db", "channel_seed"]
    paired = groups.pivot(index=match, columns="scenario", values=["total_transmitted_bits", "mean_accuracy"])
    if paired.isna().any().any() or len(paired)*2 != len(groups):
        raise AssertionError("Unpaired unicast/multicast outcomes")
    if not (paired.total_transmitted_bits.multicast <= paired.total_transmitted_bits.unicast).all():
        raise AssertionError("Protected multicast spends more than the uncoded baseline")
    result = {"passed": True, "cases_checked": len(groups), "receivers_checked": len(receivers),
              "paired_conditions": len(paired), "original_token_removal": 0,
              "max_multicast_budget_excess_bits": 0,
              "checks": ["same original information length", "K unicast copies versus one multicast waveform",
                         "coded/QPSK resource accounting", "LDPC block counts", "paired independent receiver seeds",
                         "BER denominator", "all-receiver bit-perfect success", "no budget excess"]}
    write_json(output/"verification.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="multicast_coding_budget/results/full")
    args = parser.parse_args()
    audit(args.output)
