"""Read-only source audit, targeted transport replay and fresh local QA checks."""
import ast
import importlib.metadata
import json
import math
import re
import sqlite3
import string
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bsrt_neural_compression.common import ROOT, digest, read_json, write_json
from bsrt_communication_evaluation.token_ids import decode_ids, received_ids
from bsrt_communication_evaluation.channel import qpsk_symbols, hard_qpsk, soft_qpsk
from semantic_multicast.core import stable_seed
from .transport import Channels, budget_for_context, cluster_allocation


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "results/full"
SUBSET = HERE / "results/non_singleton_receivers"
OUT = SUBSET / "verification"


def normalize(value):
    value = str(value).lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def independent_f1(prediction, reference):
    p, r = normalize(prediction).split(), normalize(reference).split()
    if not p or not r:
        return float(p == r)
    shared = sum((Counter(p) & Counter(r)).values())
    return 2.0 * shared / (len(p) + len(r))


def main():
    OUT.mkdir(exist_ok=True)
    checks = []

    def record(name, passed, details=None):
        checks.append({"check": name, "passed": bool(passed), "details": details})
        write_json(OUT / "checks.json", {"complete": False, "checks": checks})
        print(f"{'PASS' if passed else 'FAIL'}: {name}" + (f" {details}" if details is not None else ""), flush=True)

    lock = read_json(SOURCE / "run_lock.json")
    config = lock["config"]
    groups = {g["context_id"]: g for g in lock["groups"]}
    hashes = {p: digest((ROOT / p).read_bytes()) == h for p, h in lock["implementation"].items()}
    record("Locked implementation files unchanged", all(hashes.values()), hashes)
    versions = {p: importlib.metadata.version(p) for p in lock["software"]}
    record("Locked software versions match", versions == lock["software"], versions)
    source_hashes = {name: digest((SOURCE / name).read_bytes()) for name in
                     ["run_lock.json", "receiver_results.csv", "results.sqlite3"]}
    source = pd.read_csv(SOURCE / "receiver_results.csv", keep_default_na=False,
                         dtype={"context_id": str, "receiver_id": str})
    # Pandas exported list-valued references as Python literals, not JSON arrays.
    source["references"] = source.references.map(lambda value: json.dumps(ast.literal_eval(value), ensure_ascii=False))
    text_fields = {"case_id", "context_id", "receiver_id", "channel", "scenario", "question", "references",
                   "waveform_sha256", "coding", "received_context", "prediction"}
    for field in source.columns.difference(text_fields):
        source[field] = pd.to_numeric(source[field], errors="raise")
    clean = pd.read_csv(SOURCE / "clean_receiver_results.csv", keep_default_na=False,
                        dtype={"context_id": str, "receiver_id": str})
    cohort = pd.read_csv(SUBSET / "included_receivers.csv", dtype={"context_id": str, "receiver_id": str})
    eligible = {(g["context_id"], g["questions"][i]["id"])
                for g in groups.values() for c in g["clusters"] if len(c["members"]) > 1 for i in c["members"]}
    record("Eligibility matches original non-singleton cluster memberships",
           eligible == set(zip(cohort.context_id, cohort.receiver_id)) and len(eligible) == 416)
    data = source.merge(cohort, on=["context_id", "receiver_id"], how="inner", validate="many_to_one")
    keys = ["context_id", "receiver_id", "channel", "snr_db", "channel_seed", "scenario"]
    record("Complete and unique paired subgroup outcomes", len(data) == 29952 and not data.duplicated(keys).any()
           and data.groupby(["channel", "snr_db", "channel_seed", "scenario"]).size().eq(416).all())
    u = data[data.scenario.eq("unicast")].set_index(keys[:-1]).sort_index()
    m = data[data.scenario.eq("multicast")].set_index(keys[:-1]).sort_index()
    record("Identical receiver IDs, physical seeds, SNRs and source lengths across strategies",
           u.index.equals(m.index) and all(np.array_equal(u[f], m[f]) for f in
                                          ["physical_seed", "receiver_snr_db", "payload_bits"]))
    allocations = []
    physical = []
    for g in groups.values():
        b = budget_for_context(g["original_tokens"])
        full_members = [i for c in g["clusters"] for i in c["members"]]
        assert sorted(full_members) == list(range(g["k"]))
        for c in g["clusters"]:
            if len(c["members"]) > 1:
                allocations.append({"context_id": g["context_id"], "cluster_id": c["id"],
                                    **cluster_allocation(g["original_tokens"], len(c["members"]))})
        for i, q in enumerate(g["questions"]):
            offsets = np.linspace(-config["heterogeneous_span_db"] / 2, config["heterogeneous_span_db"] / 2, g["k"])
            for channel in config["channels"]:
                for center in config["snr_db_values"]:
                    for seed in config["channel_seeds"]:
                        physical.append((g["context_id"], q["id"], channel, float(center), seed,
                                         float(center + offsets[i]), stable_seed(g["context_id"], q["id"], channel,
                                         float(center), "heterogeneous", seed)))
    expected = pd.DataFrame(physical, columns=keys[:-1]+["expected_snr", "expected_seed"])
    matched = data.merge(expected, on=keys[:-1], validate="many_to_one")
    record("All receiver SNR offsets and seeds independently reconstructed",
           np.allclose(matched.receiver_snr_db, matched.expected_snr, atol=1e-14, rtol=0)
           and np.array_equal(matched.physical_seed, matched.expected_seed))
    wire = sum(a["budget_bits"] for a in allocations)
    record("Exact equal subgroup wire budget", wire == int(cohort.per_receiver_budget_bits.sum()) == 524268,
           {"per_strategy_bits": wire})
    record("Frame dimensions cover every retained source bit within budget", all(
        sum(k for k, _, _ in a["blocks"]) == a["information_bits"]
        and sum(w for _, _, w in a["blocks"]) == a["budget_bits"]
        and 10*a["budget_bits"] <= 6*a["members"]*a["original_frame_bits"]
        and all(.2 <= k/n <= .95 and w >= n for k, n, w in a["blocks"]) for a in allocations))
    record("BER and message-success labels match error counts", np.allclose(
        data.information_ber, data.information_errors/data.payload_bits, atol=1e-14, rtol=0)
        and np.array_equal(data.decode_success, data.information_errors.eq(0).astype(int)))
    connection = sqlite3.connect((SOURCE / "results.sqlite3").resolve().as_uri()+"?mode=ro", uri=True)
    case_rows = []
    for identity, group_json, receiver_json in connection.execute("SELECT id,group_json,receiver_json FROM cases"):
        group = json.loads(group_json)
        for receiver in json.loads(receiver_json):
            case_rows.append({"case_id": identity, **{f: group[f] for f in
                              ["context_id", "channel", "snr_db", "channel_seed", "scenario"]}, **receiver})
    connection.close()
    database = pd.DataFrame(case_rows).sort_values(keys).reset_index(drop=True)
    database["references"] = database.references.map(lambda value: json.dumps(value, ensure_ascii=False))
    exported = source.sort_values(keys).reset_index(drop=True)
    exported = exported[database.columns]
    pd.testing.assert_frame_equal(database, exported, check_dtype=False, check_exact=False, atol=1e-14, rtol=0)
    record("All 36,000 exported records agree with authoritative SQLite cases", True)
    f1_values, accuracy_values = [], []
    for r in data.itertuples():
        refs = json.loads(r.references)
        f1 = max(independent_f1(r.prediction, ref) for ref in refs)
        norm = normalize(r.prediction)
        correct = any(normalize(ref) == norm or normalize(ref) in norm for ref in refs)
        correct = int(correct or f1 >= .5 or r.semantic_similarity >= .75)
        f1_values.append(f1)
        accuracy_values.append(correct)
    record("Independent token-F1 recalculation for all 29,952 subgroup outcomes",
           np.allclose(f1_values, data.f1, atol=1e-14, rtol=0))
    record("Independent binary-label recalculation for all subgroup outcomes",
           np.array_equal(accuracy_values, data.accuracy))
    metrics = ["accuracy", "f1", "semantic_similarity", "information_ber", "decode_success"]
    summary = pd.read_csv(SUBSET / "summary_all_questions.csv").set_index(["channel", "snr_db", "scenario"]).sort_index()
    computed = data.groupby(["channel", "snr_db", "scenario"])[metrics].mean().sort_index()
    record("Every plotted mean reproduced from original question-level records",
           np.allclose(computed, summary[metrics], atol=1e-14, rtol=0))
    graph = pd.read_csv(SUBSET / "plots/unicast_vs_multicast_non_singleton_receivers_data.csv").set_index(
        ["channel", "snr_db", "scenario"]).sort_index()
    record("Plot data exactly match recomputed means", np.allclose(graph[metrics], computed, atol=1e-14, rtol=0))
    clean_subset = clean.merge(cohort[["context_id", "receiver_id"]], on=["context_id", "receiver_id"], validate="many_to_one")
    zero = data[data.information_errors.eq(0)].merge(clean_subset, on=["context_id", "receiver_id", "scenario"],
                                                    suffixes=("", "_clean"), validate="many_to_one")
    record("All error-free messages reproduce clean text, predictions and all QA metrics",
           all(np.array_equal(zero[f], zero[f+"_clean"]) for f in
               ["received_context", "prediction", "accuracy", "f1", "semantic_similarity"]), {"outcomes": len(zero)})

    torch.set_num_threads(2)
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
    reference = read_json(ROOT / config["qa_config"])
    tokenizer = AutoTokenizer.from_pretrained(reference["qa_model"], local_files_only=True)
    channels = Channels(config)
    print("Starting fresh transport replay (no transport/metric cache reads)...", flush=True)
    index = source.set_index(keys)
    cluster_options = [(g, c) for g in groups.values() for c in g["clusters"] if len(c["members"]) > 1]
    two = [(g, c) for g, c in cluster_options if len(c["members"]) == 2]
    chosen = [min(two, key=lambda x: x[0]["original_tokens"]), max(two, key=lambda x: x[0]["original_tokens"]),
              max(cluster_options, key=lambda x: len(x[1]["members"])),
              max(cluster_options, key=lambda x: (int(len(x[1]["members"]) > 5), x[0]["original_tokens"]))]
    specs = [(g, c, "multicast", c["members"][0]) for g, c in chosen]
    specs += [(g, c, "unicast", c["members"][0]) for g, c in chosen[:2]]
    replay = []
    for fi, (g, c, scenario, member) in enumerate(specs):
        ids = c["ids"] if scenario == "multicast" else g["individual_ids"][member]
        frame = channels.make_frame(ids, g["original_tokens"], len(c["members"]) if scenario == "multicast" else 1)
        assert np.isclose(np.mean(np.abs(frame.symbols)**2), 1., atol=1e-7)
        # Additional seeds at the noisy endpoint verify that replay is not seed-specific.
        settings = [(h, s, 42) for h in ["awgn", "fading"] for s in [0., 3., 5., 15.]]
        if fi == 2:
            settings += [(h, 0., seed) for h in ["awgn", "fading"] for seed in [123, 2024]]
        for channel, snr, seed in settings:
            qid = g["questions"][member]["id"]
            old = index.loc[(g["context_id"], qid, channel, snr, seed, scenario)]
            decoded, stats = channels.receive(frame, channel, old.receiver_snr_db, int(old.physical_seed))
            _, usable, invalid = received_ids(decoded, tokenizer)
            text = decode_ids(usable, tokenizer)
            ok = text == old.received_context and frame.fingerprint == old.waveform_sha256
            ok = ok and len(frame.information) == old.payload_bits and frame.transmitted_bits == old.frame_transmitted_bits
            ok = ok and all((pd.isna(value) and pd.isna(old[name])) or
                            (not pd.isna(value) and np.isclose(value, old[name], rtol=0, atol=1e-14))
                            for name, value in {**stats, **invalid}.items())
            replay.append({"context_id": g["context_id"], "receiver_id": qid, "scenario": scenario,
                           "members": len(c["members"]) if scenario == "multicast" else 1,
                           "channel": channel, "center_snr_db": snr, "seed": seed,
                           "actual_snr_db": old.receiver_snr_db, "passed": bool(ok),
                           "information_errors": stats["information_errors"], "stored_errors": old.information_errors})
            pd.DataFrame(replay).to_csv(OUT / "transport_replays.csv", index=False)
        print(f"Replayed source frame {fi+1}/{len(specs)}; {len(replay)} channel outcomes checked", flush=True)
    record("Fresh transport replay matches saved waveforms, received text and every recorded transport statistic",
           all(r["passed"] for r in replay), {"transmissions": len(replay)})

    print("Checking AWGN noise variance, demapper sign and theoretical raw BER...", flush=True)
    physical_checks = []
    bits = np.random.default_rng(927).integers(0, 2, size=400000, dtype=np.uint8)
    symbols = qpsk_symbols(bits)
    for snr in [0., 3., 5., 10.]:
        y, variance, gain = channels.physical(symbols, "awgn", snr, 1827)
        raw = hard_qpsk(y)
        measured = float(np.mean(raw != bits))
        theory = .5 * math.erfc(math.sqrt(10**(snr/10)/2))
        noise = y-symbols
        noise_ratio = float(np.mean(np.abs(noise)**2)/variance)
        llr = soft_qpsk(y, variance, gain)
        ok = abs(measured-theory) < .005 and abs(noise_ratio-1) < .02
        ok = ok and np.array_equal((llr > 0).astype(np.uint8), raw)
        physical_checks.append({"snr_db": snr, "measured_raw_ber": measured, "theoretical_raw_ber": theory,
                                "measured_to_expected_noise_power": noise_ratio, "passed": ok})
    pd.DataFrame(physical_checks).to_csv(OUT / "awgn_sanity.csv", index=False)
    record("AWGN power, QPSK BER and soft-demapper sign sanity checks", all(r["passed"] for r in physical_checks))

    print("Running fresh QA predictions without reading saved prediction caches...", flush=True)
    model = AutoModelForQuestionAnswering.from_pretrained(reference["qa_model"], local_files_only=True).eval()
    qa = pipeline("question-answering", model=model, tokenizer=tokenizer, device="cpu")
    probes = []
    for channel in ["awgn", "fading"]:
        for scenario in ["unicast", "multicast"]:
            for snr in [0., 15.]:
                part = data[data.channel.eq(channel) & data.scenario.eq(scenario) & data.snr_db.eq(snr)
                            & data.channel_seed.eq(42)]
                # Prefer nonempty predictions and cover corrupted as well as clean receptions.
                probe = part[part.prediction.ne("")].iloc[0].to_dict()
                with torch.inference_mode():
                    pred = qa(question=probe["question"], context=probe["received_context"],
                              handle_impossible_answer=False, max_seq_len=384, doc_stride=128, max_answer_len=15)
                probes.append({"context_id": probe["context_id"], "receiver_id": probe["receiver_id"],
                               "channel": channel, "snr_db": snr, "scenario": scenario,
                               "stored_prediction": probe["prediction"], "fresh_prediction": pred["answer"],
                               "passed": pred["answer"] == probe["prediction"]})
                print(f"Fresh QA {len(probes)}/8: {channel} {scenario} SNR={snr:g}", flush=True)
    pd.DataFrame(probes).to_csv(OUT / "fresh_qa_checks.csv", index=False)
    record("Fresh local BERT inference reproduces selected saved predictions", all(p["passed"] for p in probes), {"predictions": len(probes)})
    del model, qa

    from sentence_transformers import SentenceTransformer
    print("Checking fresh semantic-similarity scores on selected and threshold-near answers...", flush=True)
    near = data.assign(gap=(data.semantic_similarity-.75).abs()).sort_values("gap").head(12)
    examples = pd.concat([near, data[data.channel_seed.eq(42)].groupby(["channel", "snr_db", "scenario"]).head(1)])
    examples = examples.drop_duplicates(["prediction", "references"])
    strings = sorted(set(examples.prediction) | {r for refs in examples.references for r in json.loads(refs)})
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True, device="cpu")
    vectors = similarity_model.encode(strings, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
    embeddings = dict(zip(strings, vectors, strict=True))
    similarities = []
    for r in examples.itertuples():
        fresh = max(float(embeddings[r.prediction] @ embeddings[ref]) for ref in json.loads(r.references))
        similarities.append({"prediction": r.prediction, "references": r.references, "stored_similarity": r.semantic_similarity,
                             "fresh_similarity": fresh, "absolute_difference": abs(fresh-r.semantic_similarity),
                             "threshold_decision_matches": bool((fresh >= .75) == (r.semantic_similarity >= .75))})
    pd.DataFrame(similarities).to_csv(OUT / "fresh_similarity_checks.csv", index=False)
    record("Fresh Sentence-BERT scores and threshold decisions reproduce selected records",
           all(s["absolute_difference"] < 2e-5 and s["threshold_decision_matches"] for s in similarities),
           {"answer_pairs": len(similarities), "max_absolute_error": max(s["absolute_difference"] for s in similarities)})
    record("Original run lock, raw CSV and SQLite source files unchanged by audit",
           all(digest((SOURCE / name).read_bytes()) == h for name, h in source_hashes.items()))
    passed = all(c["passed"] for c in checks)
    write_json(OUT / "checks.json", {"complete": True, "passed": passed, "checks": checks,
                                      "source_sha256": source_hashes})
    lines = ["# Verification of the latest non-singleton multicast comparison", "",
             f"Overall outcome: **{'all checks passed' if passed else 'one or more checks failed; inspect details'}**.", "",
             "Scope: 416 questions, 30 contexts, 109 non-singleton clusters; 29,952 paired receiver outcomes.",
             "Source records were read only. Replays did not consume the experimental transport/QA caches.", "",
             "## Checks", "", "| Check | Result |", "|---|---|"]
    lines.extend(f"| {c['check']} | {'PASS' if c['passed'] else 'FAIL'} |" for c in checks)
    lines += ["", "## Interpretation and limitations", "",
              "The full record checks cover cohort membership, aggregate budgets, pairing, error denominators, all F1/accuracy labels, plot means and error-free clean consistency.",
              f"Fresh transport replay covers {len(replay)} selected transmissions, not all 29,952 outcomes. Fresh QA covers eight predictions; fresh similarity checks cover {len(similarities)} answer/reference pairs.",
              "A successful replay is evidence of reproducibility, not an independent physical-channel implementation or proof that the model matches a real network.",
              "All retained bits enter LDPC for included multicast clusters; source bits are used to compute evaluation error statistics, not as decoder inputs. Soft demapping uses received symbols and known channel/noise parameters.",
              "The 0 dB x coordinate is a group center, with original receiver offsets within -3 to +3 dB. Code rates vary by cluster size and large clusters use repetition. High low-SNR QA is not a claim of universally error-free transport.",
              "This remains a supervised task-aware, gold-answer-gradient reference, not BSRT or question-blind compression. Binary correctness uses permissive historical thresholds and is not manually verified factual correctness.",
              "TDL is normalized/static with ideal CSI and approximate equalizer LLR variance. Headers, signaling, pilots, CRC, latency, energy at receivers and practical multicast feedback are not modeled.",
              "Equal aggregate transmitted bits are verified; equal per-receiver energy or performance against all possible coded-unicast allocations is not established.",
              "Excluding singletons conditions on a selected subgroup; it must not be reported as full-500 performance. No new significance claim is made.", "",
              "## Evidence", "", "- `checks.json`: results and locked source hashes.",
              "- `transport_replays.csv`: exact replay comparisons.", "- `awgn_sanity.csv`: measured/theoretical physical sanity checks.",
              "- `fresh_qa_checks.csv`: uncached BERT predictions.", "- `fresh_similarity_checks.csv`: fresh Sentence-BERT comparisons."]
    (OUT / "VERIFICATION_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(OUT / "VERIFICATION_REPORT.md", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
