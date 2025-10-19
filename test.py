#!/usr/bin/env python
"""
Quick smoke test for unsloth/gpt-oss-20b 4-bit inference with streaming output.
"""

from __future__ import annotations

import sys

import torch
from transformers import TextStreamer


def main() -> None:
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        raise SystemExit("A CUDA or MPS device is required to run unsloth/gpt-oss-20b.")

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:  # pragma: no cover - convenience message
        raise SystemExit(
            "This script requires the `unsloth` package. Install it with `pip install unsloth`."
        ) from exc
    except NotImplementedError as exc:  # pragma: no cover - raised on unsupported hardware
        raise SystemExit(str(exc)) from exc

    device = "cuda" if torch.cuda.is_available() else "mps"
    model_name = "unsloth/gpt-oss-20b"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=5000,
        load_in_4bit=True,
        offload_embedding=True,
    )
    model.config.use_cache = True
    model.generation_config.use_cache = True
    model.eval()

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    chat_prompt = [
        {"role": "system", "content": "You are a concise assistant. Reply briefly unless the user asks for details."},
        {"role": "user", "content": "量子化された20Bモデルでの推論がどれくらい速いか、手短に教えて。"},
    ]
    inputs = tokenizer.apply_chat_template(
        chat_prompt,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    print(f"--- Streaming response from {model_name} ({device}, 4-bit) ---\n", flush=True)

    with torch.inference_mode():
        model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )
    print("\n--- Done ---")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        sys.stderr.write(str(exc) + "\n")
        sys.exit(exc.code)
