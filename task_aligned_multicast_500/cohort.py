"""All final 500 questions, full contexts and task-alignment clustering."""
import importlib.metadata
import json

import numpy as np
import pandas as pd

from bsrt_neural_compression.common import ROOT, digest, read_json, resolve, write_json
from paper_multi_receiver_task_alignment.run_task_alignment import fit_task_alignment, task_distortion_gradient_l2
from semantic_multicast.core import select_ids
from .transport import budget_for_context, cluster_allocation


def prepare(config, smoke=False):
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    output = resolve(config["output_dir"])
    if not output.resolve().is_relative_to((ROOT/"task_aligned_multicast_500/results").resolve()):
        raise ValueError("Use this experiment's separate results directory")
    output.mkdir(parents=True, exist_ok=True)
    reference = read_json(resolve(config["qa_config"]))
    tokenizer = AutoTokenizer.from_pretrained(reference["qa_model"], local_files_only=True)
    records = pd.read_csv(resolve(config["receiver_csv"]),
                          usecols=["dataset_index", "question", "original_context", "true_answers"])
    records = records.drop_duplicates("dataset_index").set_index("dataset_index")
    if len(records) != 500:
        raise ValueError("Expected the final 500-question evaluation")
    paths = sorted(resolve(config["relevance_cache_dir"]).glob("*.npz"))
    source_ids = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            source_ids.extend(map(int, data["receiver_ids"]))
    if len(source_ids) != 500 or set(source_ids) != set(map(int, records.index)):
        raise ValueError("Gradient caches do not cover exactly the final 500 questions")
    if smoke:
        paths = paths[:1]
    groups, provenance, allocations = [], [], []
    model = None
    recomputed_questions = reused_extended_questions = 0
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            context_id = str(data["context_id"].item())
            context = str(data["original_context"].item())
            if str(data["importance_estimator"].item()) != "distortion_gradient_l2":
                raise ValueError("Unexpected importance estimator")
            ids = tokenizer.encode(context, add_special_tokens=False, truncation=False)
            receiver_ids = list(map(int, data["receiver_ids"]))
            questions = []
            for i, qid in enumerate(receiver_ids):
                row = records.loc[qid]
                refs = json.loads(str(data["true_answers"][i]))
                if row.original_context != context or row.question != str(data["questions"][i]) or json.loads(row.true_answers) != refs:
                    raise ValueError("Question/context/reference differs from final records")
                questions.append({"id": str(qid), "question": str(row.question), "answers": {"text": refs}})
            qmap = {q["id"]: q for q in questions}
            source_hash = digest(path.read_bytes())
            used_source = {"source": str(path.relative_to(ROOT)), "sha256": source_hash,
                           "original_tokens": len(ids), "cached_tokens": len(data["token_ids"])}
            if np.array_equal(data["token_ids"], ids):
                relevance = data["relevance"].astype(float)
            else:
                cache = output/"full_length_relevance"/(context_id+".npz")
                signature = digest([source_hash, digest(ids), questions, reference,
                                    digest((ROOT/"paper_multi_receiver_task_alignment/run_task_alignment.py").read_bytes())])
                if cache.exists():
                    with np.load(cache, allow_pickle=False) as saved:
                        if str(saved["signature"].item()) != signature:
                            raise ValueError("Extended cache provenance changed")
                        relevance = saved["relevance"]
                else:
                    reused = {}
                    earlier = resolve("multicast_coding_budget/results/compressed40/full_length_relevance")/(context_id+".npz")
                    if earlier.exists():
                        with np.load(earlier, allow_pickle=False) as partial:
                            prior_ids = partial["receiver_ids"].tolist()
                            prior_signature = digest([source_hash, digest(ids), [qmap[q] for q in prior_ids], reference,
                                digest((ROOT/"paper_multi_receiver_task_alignment/run_task_alignment.py").read_bytes())])
                            if str(partial["signature"].item()) == prior_signature:
                                reused = dict(zip(prior_ids, partial["relevance"], strict=True))
                                reused_extended_questions += len(reused)
                    if model is None:
                        model = AutoModelForQuestionAnswering.from_pretrained(reference["qa_model"], local_files_only=True)
                        model.eval()
                    canonical = tokenizer.decode(ids, skip_special_tokens=True)
                    values = []
                    for q in questions:
                        if q["id"] in reused:
                            values.append(reused[q["id"]])
                            continue
                        paired = tokenizer(q["question"], canonical, truncation="only_second", max_length=512)
                        if sum(s == 1 for s in paired.sequence_ids()) != len(ids):
                            raise ValueError("Full gradient context would truncate; stop rather than substitute")
                        values.append(task_distortion_gradient_l2(q["question"], q["answers"]["text"], canonical, ids, tokenizer, model))
                        recomputed_questions += 1
                    relevance = np.asarray(values)
                    cache.parent.mkdir(exist_ok=True, parents=True)
                    np.savez_compressed(cache, signature=np.asarray(signature), relevance=relevance)
                    print(f"Full-length relevance {context_id}: {len(ids)} tokens, {len(questions)} questions; "
                          f"reused {len(reused)}, calculated {len(questions)-len(reused)}", flush=True)
                used_source.update(extended_cache=str(cache.relative_to(ROOT)), extended_sha256=digest(cache.read_bytes()))
        if relevance.shape != (len(questions), len(ids)) or not np.isfinite(relevance).all():
            raise ValueError("Malformed full-context relevance")
        alignment = fit_task_alignment(relevance/15., config)
        labels = alignment["selected"]["labels"].astype(int)
        priorities = alignment["selected"]["priorities"]
        keep = budget_for_context(len(ids))["kept_tokens"]
        clusters = []
        for label, priority in priorities.items():
            members = [i for i, x in enumerate(labels) if x == label]
            selected = select_ids(ids, priority, keep)
            clusters.append({"id": int(label), "members": members, "ids": selected})
            allocations.append({"context_id": context_id, "cluster_id": int(label),
                                "receiver_ids": [questions[i]["id"] for i in members],
                                **cluster_allocation(len(ids), len(members))})
        group = {"context_id": context_id, "id": digest([context_id, [q["id"] for q in questions]])[:20],
                 "questions": questions, "k": len(questions), "original_tokens": len(ids),
                 "individual_ids": [select_ids(ids, row, keep) for row in relevance], "clusters": clusters}
        groups.append(group)
        provenance.append(used_source)
    del model
    if not smoke and (len(groups) != 36 or sum(g["k"] for g in groups) != 500):
        raise ValueError("The full cohort must have 36 contexts and 500 questions")
    source_paths = ["task_aligned_multicast_500/cohort.py", "task_aligned_multicast_500/transport.py",
                    "task_aligned_multicast_500/run.py", "paper_multi_receiver_task_alignment/run_task_alignment.py",
                    "multicast_coding_budget/compressed_transport.py", "multicast_coding_budget/transport.py",
                    "semantic_multicast/core.py", "bsrt_budgeted_ldpc/channel.py", "bsrt_communication_evaluation/channel.py",
                    "bsrt_communication_evaluation/token_ids.py", "bsrt_neural_compression/reward.py", "main.py"]
    lock = {"config": config, "groups": groups, "provenance": provenance,
            "qa_config_sha256": digest(resolve(config["qa_config"]).read_bytes()),
            "implementation": {p: digest((ROOT/p).read_bytes()) for p in source_paths},
            "software": {p: importlib.metadata.version(p) for p in ["torch", "sionna", "numpy", "transformers", "scikit-learn"]}}
    lock_path = output/"run_lock.json"
    if lock_path.exists() and read_json(lock_path) != lock:
        raise ValueError("Source, configuration or cohort changed: use a new output directory")
    write_json(lock_path, lock)
    conditions = len(config["channels"])*len(config["snr_db_values"])*len(config["channel_seeds"])
    plan = {"contexts": len(groups), "questions": sum(g["k"] for g in groups),
            "receivers_per_context_min": min(g["k"] for g in groups),
            "receivers_per_context_max": max(g["k"] for g in groups),
            "clusters": sum(len(g["clusters"]) for g in groups),
            "singleton_clusters": sum(len(c["members"]) == 1 for g in groups for c in g["clusters"]),
            "expected_cases": len(groups)*conditions*2,
            "expected_receiver_outcomes": sum(g["k"] for g in groups)*conditions*2,
            "removal_target_percent": 40, "budget_fraction": .6,
            "averaging": "equal weight to each of the 500 questions, then average channel seeds",
            "question_blind": False, "references_used_for_task_gradients": True,
            "gradient_questions_calculated_this_invocation": recomputed_questions,
            "extended_gradient_questions_reused_this_invocation": reused_extended_questions}
    write_json(output/"budget_allocations.json", allocations)
    write_json(output/"plan.json", plan)
    return groups, plan, reference
