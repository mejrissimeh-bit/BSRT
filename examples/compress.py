"""Compress one context with the released BSRT retention head."""

from pathlib import Path

from bsrt_neural_compression import BSRT


context = (
    "Glaucoma is a progressive optic neuropathy. Early detection and regular "
    "monitoring can help preserve vision by enabling timely treatment."
)
root = Path(__file__).resolve().parents[1]
model = BSRT.from_portfolio(root / "checkpoints")

print("Original:", context)
print("Compressed:", model.compress(context, removal_ratio=0.40))
