from __future__ import annotations

import numpy as np

from .common import digest, read_json, write_json
from .model import budget_mask, render_context


def refine_swaps(mask, scores, groups, rng):
    """Swap complete equal-sized WordPiece groups, preserving exact mask size."""
    values = scores + rng.gumbel(size=len(scores)) * .5
    kept = [group for group in groups if mask[group].all()]
    removed = [group for group in groups if not mask[group].any()]
    rng.shuffle(kept)
    for outgoing in kept:
        options = [g for g in removed if len(g) == len(outgoing)]
        if options:
            incoming = max(options, key=lambda group: float(values[group].mean()))
            result = mask.copy()
            result[outgoing], result[incoming] = False, True
            return result
    return mask.copy()


def optimize_mask(policy, record, questions, percent, qa, config, path, round_index):
    questions_fingerprint = digest(questions)
    if path.exists():
        cached = read_json(path)
        if cached["questions_fingerprint"] != questions_fingerprint or cached["config_fingerprint"] != config["config_fingerprint"]:
            raise ValueError("Search-cache provenance mismatch")
        return cached
    ratio = percent / 100
    scores, feature = policy.scores(record["context"], ratio)
    rng = np.random.default_rng(int(digest([record["context_id"], percent, round_index, config["seed"]])[:12], 16))
    distribution = scores.copy()
    evaluated = {}
    proposed = [budget_mask(scores, feature["groups"], ratio)]
    for iteration in range(config["search_iterations"]):
        while len(proposed) < config["search_population"]:
            proposed.append(budget_mask(distribution + rng.gumbel(size=len(scores))*config["search_temperature"], feature["groups"], ratio))
        for mask in proposed:
            key = digest(mask.astype(int).tolist())
            if key not in evaluated:
                compressed = render_context(record["context"], feature["offsets"], mask)
                evaluated[key] = {"mask": mask.astype(int).tolist(), **qa.reward(compressed, questions)}
        elites = sorted(evaluated.values(), key=lambda item: item["f1"], reverse=True)[:config["search_elites"]]
        probability = np.asarray([item["mask"] for item in elites]).mean(axis=0).clip(.05, .95)
        distribution = .4*scores + .6*np.log(probability/(1-probability))
        best = np.asarray(elites[0]["mask"], dtype=bool)
        proposed = [best, refine_swaps(best, distribution, feature["groups"], rng)]
    ranked = sorted(evaluated.values(), key=lambda item: item["f1"], reverse=True)
    elite = ranked[:config["search_elites"]]
    rewards = np.asarray([item["f1"] for item in elite])
    weights = np.exp((rewards-rewards.max())/config["reward_temperature"])
    weights /= weights.sum()
    target = np.average(np.asarray([item["mask"] for item in elite]), axis=0, weights=weights)
    result = {"context_id": record["context_id"], "percent": percent, "round": round_index,
              "questions_fingerprint": questions_fingerprint, "config_fingerprint": config["config_fingerprint"],
              "target": target.tolist(), "weight": max(.05, float(rewards.max())),
              "best_reward": float(rewards.max()), "initial_reward": next(iter(evaluated.values()))["f1"],
              "evaluated_masks": len(evaluated), "elites": elite}
    write_json(path, result)
    return result
