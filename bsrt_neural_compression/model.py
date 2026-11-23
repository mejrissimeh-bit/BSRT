from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer

from .common import digest, read_json, resolve


class RetentionNetwork(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=96, layers=2, heads=4, dropout=.1,
                 contextual=True, budget_conditioning=True, structural_features=True,
                 contextual_vectors=True, position_embedding=True,
                 disabled_structural_features=()):
        super().__init__()
        self.contextual = contextual
        self.budget_conditioning = budget_conditioning
        self.structural_features = structural_features
        self.contextual_vectors = contextual_vectors
        self.position_embedding = position_embedding
        disabled = tuple(disabled_structural_features)
        if len(set(disabled)) != len(disabled) or any(type(i) is not int or not 0 <= i < 8 for i in disabled):
            raise ValueError("Disabled structural features must be unique indices from 0 to 7")
        self.disabled_structural_features = disabled
        feature_mask = torch.ones(8)
        if disabled:
            feature_mask[list(disabled)] = 0
        self.register_buffer("structural_input_mask", feature_mask, persistent=False)
        self.project = nn.Linear(input_dim, hidden_dim)
        self.structure = nn.Linear(8, hidden_dim)
        self.budget = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.positions = nn.Embedding(512, hidden_dim)
        if contextual:
            layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim*3, dropout, batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        else:
            self.transformer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*3), nn.GELU(), nn.Linear(hidden_dim*3, hidden_dim))
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        if not budget_conditioning:
            self.budget.requires_grad_(False)
        if not structural_features:
            self.structure.requires_grad_(False)
        if not contextual_vectors:
            self.project.requires_grad_(False)
        if not position_embedding:
            self.positions.requires_grad_(False)

    def forward(self, contextual_embeddings, structure, removal_ratio):
        if contextual_embeddings.ndim != 2:
            raise ValueError("One context at a time: shape must be [tokens, hidden]")
        positions = torch.arange(len(contextual_embeddings), device=contextual_embeddings.device).clamp(max=511)
        hidden = (self.project(contextual_embeddings) if self.contextual_vectors else
                  contextual_embeddings.new_zeros((len(contextual_embeddings), self.project.out_features)))
        if self.structural_features:
            if self.disabled_structural_features:
                structure = structure * self.structural_input_mask
            hidden = hidden + self.structure(structure)
        if self.position_embedding:
            hidden = hidden + self.positions(positions)
        if self.budget_conditioning:
            budget = hidden.new_tensor([removal_ratio, 1-removal_ratio])
            hidden = hidden + self.budget(budget)
        hidden = self.transformer(hidden.unsqueeze(0))
        return self.out(hidden).squeeze(0).squeeze(-1)


def lexical_groups(tokenizer, ids, offsets):
    """Keep WordPiece units intact without requiring the original questions."""
    groups = []
    tokens = tokenizer.convert_ids_to_tokens(ids)
    for index, (token, offset) in enumerate(zip(tokens, offsets)):
        if token.startswith("##") and groups and offset[0] == offsets[index-1][1]:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def budget_mask(scores, groups, removal_ratio):
    """Maximum-score whole-WordPiece subset, nearest feasible token budget.

    Knapsack decisions keep all continuation pieces with their root. Most
    contexts contain singleton groups, so the floored token budget is exact.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if not 0 <= removal_ratio < 1 or not np.isfinite(scores).all():
        raise ValueError("Finite scores and 0 <= removal_ratio < 1 are required")
    n = len(scores)
    keep = n - int(n * removal_ratio)
    if n == 0:
        return np.zeros(0, dtype=bool)
    dp = np.full(n+1, -np.inf)
    dp[0] = 0.0
    decisions = np.zeros((len(groups), n+1), dtype=bool)
    for i, group in enumerate(groups):
        size = len(group)
        value = float(scores[group].sum())
        candidate = dp[:-size] + value
        better = candidate > dp[size:]
        dp[size:] = np.maximum(dp[size:], candidate)
        decisions[i, size:] = better
    feasible = np.flatnonzero(np.isfinite(dp))
    count = min(feasible.tolist(), key=lambda value: (abs(value-keep), -value))
    result = np.zeros(n, dtype=bool)
    for i in range(len(groups)-1, -1, -1):
        if decisions[i, count]:
            result[groups[i]] = True
            count -= len(groups[i])
    return result


def render_context(context, offsets, mask):
    # Original character slices retain spelling and punctuation; re-tokenized
    # token counts are still measured independently in evaluation.
    spans = [(a, b) for (a, b), keep in zip(offsets, mask) if keep]
    pieces = []
    previous_end = None
    for start, end in spans:
        if previous_end is not None and start > previous_end:
            gap = context[previous_end:start]
            pieces.append(gap if gap.isspace() else " ")
        pieces.append(context[start:end])
        previous_end = end
    return "".join(pieces).strip()


class ContextFeatures:
    def __init__(self, config, qa_model, tokenizer):
        self.config, self.qa_model, self.tokenizer = config, qa_model, tokenizer
        self.device = next(qa_model.parameters()).device
        encoder_fingerprint = {"model": config["qa_model"], "commit": getattr(qa_model.config, "_commit_hash", None), "format": 1}
        self.cache_dir = resolve(config["cache_dir"]) / ("embeddings_" + digest(encoder_fingerprint)[:12])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, context, use_cache=True):
        path = self.cache_dir / (digest(context) + ".pt")
        if use_cache and path.exists():
            return torch.load(path, map_location="cpu", weights_only=False)
        encoded = self.tokenizer(context, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        hidden = []
        with torch.inference_mode():
            for start in range(0, len(ids), 510):
                part = ids[start:start+510]
                inputs = torch.tensor([[self.tokenizer.cls_token_id, *part, self.tokenizer.sep_token_id]], device=self.device)
                hidden.append(self.qa_model.base_model(input_ids=inputs).last_hidden_state[0, 1:-1].cpu().half())
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        frequency = Counter(ids)
        structure = [[i/max(len(ids)-1, 1), len(t)/20, float(t.startswith("##")), float(any(c.isdigit() for c in t)),
                      float(t[:1].isupper()), float(not any(c.isalnum() for c in t)), frequency[token_id]/len(ids),
                      float(i == 0 or "." in context[offsets[i-1][1]:offsets[i][0]])]
                     for i, (t, token_id) in enumerate(zip(tokens, ids))]
        value = {"input_ids": ids, "offsets": offsets, "embeddings": torch.cat(hidden),
                 "structure": torch.tensor(structure, dtype=torch.float32), "groups": lexical_groups(self.tokenizer, ids, offsets)}
        if use_cache:
            temporary = path.with_suffix(".tmp")
            torch.save(value, temporary)
            temporary.replace(path)
        return value


class BSRT:
    """Deployable inference accepts only the context and target removal ratio."""
    def __init__(self, network, features):
        self.network, self.features = network, features
        self.network.eval()

    def scores(self, context, removal_ratio):
        feature = self.features.get(context)
        device = next(self.network.parameters()).device
        with torch.inference_mode():
            values = self.network(feature["embeddings"].to(device).float(), feature["structure"].to(device), removal_ratio)
        return values.cpu().numpy(), feature

    def select(self, context, removal_ratio):
        scores, feature = self.scores(context, removal_ratio)
        return budget_mask(scores, feature["groups"], removal_ratio), feature

    def compress(self, context, removal_ratio=0.40):
        if not isinstance(context, str):
            raise TypeError("context must be a string")
        if not 0 <= removal_ratio < 1:
            raise ValueError("removal_ratio must be in [0, 1)")
        if not context.strip():
            return context
        mask, feature = self.select(context, removal_ratio)
        return render_context(context, feature["offsets"], mask)

    @classmethod
    def from_pretrained(cls, checkpoint_path, cache_dir=None):
        """Load a trusted legacy training checkpoint.

        Legacy ``.pt`` files use Python pickle. For public distribution, prefer
        :meth:`from_portfolio`, which reads Safetensors weights and JSON only.
        """
        from transformers import AutoModelForQuestionAnswering
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = saved["config"]
        if cache_dir is not None:
            config["cache_dir"] = str(cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(config["qa_model"], local_files_only=config["local_files_only"])
        encoder = AutoModelForQuestionAnswering.from_pretrained(config["qa_model"], local_files_only=config["local_files_only"])
        encoder.eval().requires_grad_(False)
        network = RetentionNetwork(**config["model"])
        network.load_state_dict(saved["state_dict"])
        return cls(network, ContextFeatures(config, encoder, tokenizer))

    @classmethod
    def from_portfolio(cls, checkpoint_dir, cache_dir=None, local_files_only=False):
        """Build BSRT from the portable JSON and Safetensors release files."""
        from safetensors.torch import load_file
        from transformers import AutoModelForQuestionAnswering

        checkpoint_dir = resolve(checkpoint_dir)
        config = read_json(checkpoint_dir / "config.json")
        config["local_files_only"] = bool(local_files_only)
        if cache_dir is not None:
            config["cache_dir"] = str(cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            config["qa_model"], local_files_only=config["local_files_only"]
        )
        encoder = AutoModelForQuestionAnswering.from_pretrained(
            config["qa_model"], local_files_only=config["local_files_only"]
        )
        encoder.eval().requires_grad_(False)
        network = RetentionNetwork(**config["model"])
        network.load_state_dict(load_file(checkpoint_dir / config["weights"]))
        return cls(network, ContextFeatures(config, encoder, tokenizer))
