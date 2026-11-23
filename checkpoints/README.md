# BSRT-A0 checkpoint

`bsrt_a0.safetensors` is the trained retention head selected by validation F1. It contains 37 tensors and 320,353 parameters. It does not contain the frozen BERT encoder, optimizer state, raw data, or executable Python objects.

The matching architecture and encoder identifier are in `config.json`. The encoder is downloaded from Hugging Face on the first online demo run.
