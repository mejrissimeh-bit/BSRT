from __future__ import annotations

import json
import sqlite3
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

from .common import ROOT, digest, resolve

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from metrics_reference import exact_match_score, f1_score, evaluate_prediction


class QAReward:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config["qa_model"], local_files_only=config["local_files_only"])
        self.model = AutoModelForQuestionAnswering.from_pretrained(config["qa_model"], local_files_only=config["local_files_only"])
        self.model.to(self.device).eval().requires_grad_(False)
        self.qa = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer, device=self.device)
        cache_dir = resolve(config["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_dir / "qa_rewards.sqlite3")
        self.db.execute("CREATE TABLE IF NOT EXISTS predictions (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.calls = 0
        self.seconds = 0.0
        self.similarity_model = None
        self.settings = {"checkpoint": config["qa_model"], "commit": getattr(self.model.config, "_commit_hash", None),
                         "max_seq_len": 384, "doc_stride": 128, "max_answer_len": 15, "schema": 1}

    def predictions(self, context, questions, allow_no_answer=False):
        settings = {**self.settings, "allow_no_answer": allow_no_answer}
        result, missing = {}, []
        for question in questions:
            key = digest([settings, context, question["question"]])
            cached = self.db.execute("SELECT value FROM predictions WHERE key=?", (key,)).fetchone()
            if cached:
                result[question["id"]] = json.loads(cached[0])
            else:
                missing.append((key, question))
        for start in range(0, len(missing), self.config["qa_batch_size"]):
            chunk = missing[start:start+self.config["qa_batch_size"]]
            inputs = [{"context": context, "question": q["question"]} for _, q in chunk]
            begin = time.perf_counter()
            if not context.strip():
                values = [{"answer": "", "score": 0.0} for _ in inputs]
            else:
                with torch.inference_mode():
                    values = self.qa(inputs, batch_size=self.config["qa_batch_size"], handle_impossible_answer=allow_no_answer,
                                     max_seq_len=384, doc_stride=128, max_answer_len=15)
                if isinstance(values, dict):
                    values = [values]
            self.seconds += time.perf_counter()-begin
            for (key, question), value in zip(chunk, values, strict=True):
                value = {"answer": value["answer"], "score": float(value["score"])}
                self.db.execute("INSERT OR REPLACE INTO predictions VALUES (?,?)", (key, json.dumps(value)))
                result[question["id"]] = value
                self.calls += 1
            self.db.commit()
        return [result[q["id"]] for q in questions]

    def reward(self, context, questions):
        predictions = self.predictions(context, questions, allow_no_answer=True)
        f1s, answerable, impossible = [], [], []
        for question, prediction in zip(questions, predictions, strict=True):
            references = question["answers"]["text"] or [""]
            score = max(f1_score(prediction["answer"], ref) for ref in references)
            f1s.append(score)
            (answerable if question["answers"]["text"] else impossible).append(score)
        return {"f1": float(np.mean(f1s)), "answerable_f1": float(np.mean(answerable)) if answerable else None,
                "unanswerable_f1": float(np.mean(impossible)) if impossible else None, "questions": len(f1s)}

    def metrics(self, context, questions, allow_no_answer=False, semantic=False):
        predictions = self.predictions(context, questions, allow_no_answer)
        if semantic and self.similarity_model is None:
            from sentence_transformers import SentenceTransformer
            self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True, device="cpu")
        rows = []
        for question, prediction in zip(questions, predictions, strict=True):
            refs = question["answers"]["text"]
            if refs and semantic:
                acc, em, f1, sim = evaluate_prediction(prediction["answer"], refs, self.similarity_model)
            else:
                em = max(exact_match_score(prediction["answer"], ref) for ref in refs or [""])
                f1 = max(f1_score(prediction["answer"], ref) for ref in refs or [""])
                # The previous metric has max([]) and empty-substring problems
                # for SQuAD2 no-answer items. Exact abstention is explicit here.
                acc = int(em if not refs else f1 >= .5)
                sim = float(em) if not refs else None
            rows.append({"question_id": question["id"], "question": question["question"], "references": refs,
                         "prediction": prediction["answer"], "answerable": bool(refs), "accuracy": int(acc),
                         "exact_match": float(em), "f1": float(f1), "semantic_similarity": sim})
        return rows

    def close(self):
        self.db.close()
