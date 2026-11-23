from __future__ import annotations

import copy
import subprocess
import sys

from .common import ROOT, digest, resolve, write_json


def run_ablations(config):
    variants = {
        "no_answer_span_pretraining": {"pretrain_epochs": 0},
        "no_mask_distillation": {"improvement_rounds": 0},
        "single_question_rewards": {"max_train_questions": 1},
        "one_improvement_round": {"improvement_rounds": 1},
        "no_contextual_scoring": {"contextual": False},
        "no_budget_conditioning": {"budget_conditioning": False},
    }
    for name, changes in variants.items():
        if name == "no_budget_conditioning" and len(config["removal_percentages"]) == 1:
            print("Budget-conditioning ablation requires multiple budgets; skipped for 40%-only pilot.", flush=True)
            continue
        changed = copy.deepcopy(config)
        changed.pop("config_fingerprint", None)
        changed["output_dir"] = str(resolve(config["output_dir"]) / "ablations" / name)
        for key, value in changes.items():
            if key in changed["model"]:
                changed["model"][key] = value
            else:
                changed[key] = value
        path = resolve(changed["output_dir"]) / "config.json"
        write_json(path, changed)
        subprocess.run([sys.executable, "-m", "bsrt_neural_compression.run", "pilot", "--config", str(path)], cwd=ROOT, check=True)
