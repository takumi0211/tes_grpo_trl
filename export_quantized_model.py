#!/usr/bin/env python
"""
Merge the trained LoRA adapter into the GPT-OSS base model and store a
quantization hint (MXFP4) alongside the merged weights.

Adjust the constants below if paths differ, then run:
    python export_quantized_model.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

# ----- Export parameters -----
BASE_MODEL_ID = "openai/gpt-oss-20b"
ADAPTER_PATH = Path("runs_10step分/grpo_gptoss20b_lora4_tes")
OUTPUT_DIR = Path("exports/grpo_gptoss20b_lora4_tes_merged")
SAFE_SERIALIZATION = True  # set False to save as PyTorch binaries (.bin)


def _resolve_tokenizer_source(adapter_dir: Path, fallback_model: str) -> str | Path:
    tok_config = adapter_dir / "tokenizer_config.json"
    return adapter_dir if tok_config.exists() else fallback_model


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer_source = _resolve_tokenizer_source(ADAPTER_PATH, BASE_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    tokenizer.save_pretrained(OUTPUT_DIR)

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        load_kwargs["attn_implementation"] = "kernels-community/vllm-flash-attn3"

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **load_kwargs)
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, torch_dtype=torch.bfloat16)

    merged_model = peft_model.merge_and_unload()
    quant_cfg = Mxfp4Config(dequantize=False, compute_dtype=torch.bfloat16)
    merged_model.config.quantization_config = quant_cfg.to_dict()
    merged_model.save_pretrained(OUTPUT_DIR, safe_serialization=SAFE_SERIALIZATION)

    (OUTPUT_DIR / "quantization_config.json").write_text(quant_cfg.to_json_string(), encoding="utf-8")

    adapter_config_src = ADAPTER_PATH / "adapter_config.json"
    if adapter_config_src.exists():
        adapter_cfg = json.loads(adapter_config_src.read_text(encoding="utf-8"))
        adapter_cfg_path = OUTPUT_DIR / "adapter_config.merged.json"
        adapter_cfg_path.write_text(json.dumps(adapter_cfg, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
