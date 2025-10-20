#!/usr/bin/env python
"""
Quick smoke test for unsloth/gpt-oss-20b 4-bit inference with streaming output.
"""

from __future__ import annotations

import importlib.util
import sys

import torch


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

    try:
        from transformers import TextStreamer
    except ImportError as exc:  # pragma: no cover - convenience message
        raise SystemExit(
            "This script requires the `transformers` package. Install it with `pip install transformers`."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "mps"
    model_name = "unsloth/gpt-oss-20b"

    use_flash_attention = any(
        importlib.util.find_spec(name) is not None
        for name in ("flash_attn", "flash_attn_2_cuda")
    )
    fa_kwargs = {}
    if use_flash_attention:
        fa_kwargs["attn_implementation"] = "flash_attention_2"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=5000,
        load_in_4bit=True,
        offload_embedding=False,
        **fa_kwargs,
    )
    if use_flash_attention:
        set_attn = getattr(model, "set_attn_implementation", None)
        if callable(set_attn):
            try:
                set_attn("flash_attention_2")
            except Exception as exc:  # pragma: no cover - diagnostic
                sys.stderr.write(f"Could not enable FlashAttention 2: {exc}\n")
        else:
            sys.stderr.write("FlashAttention 2 support unavailable on this model build.\n")
    else:
        sys.stderr.write(
            "FlashAttention 2 not detected. Install the `flash-attn` package to enable it.\n"
        )

    FastLanguageModel.for_inference(model)
    model.config.use_cache = True
    model.generation_config.use_cache = True
    model.eval()

    target_device = torch.device(device)
    if hasattr(model, "get_input_embeddings"):
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is not None and hasattr(input_embeddings, "to"):
            input_embeddings.to(target_device)
    if hasattr(model, "get_output_embeddings"):
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and hasattr(output_embeddings, "to"):
            output_embeddings.to(target_device)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    chat_prompt = [
        {
            "role": "system",
            "content": (
                "You are an optimisation agent that supervises a thermal energy storage (TES) plant. Your job is to minimise cumulative CO₂ emissions over the full simulation horizon by planning TES charge and discharge decisions.\n"
                "Developer instructions: respond using ASCII characters only. Return a single line formatted exactly as `[action_index]`, where action_index is an integer in {0, 1, 2, 3}. Do not include additional text, explanations, markdown, or keys."
            ),
        },
        {
            "role": "user",
            "content": (
                "Objective:\n"
                "- Minimise total CO₂ emissions = electricity consumption × time-varying CO₂ intensity over the horizon.\n"
                "\n"
                "Current context:\n"
                "- Current time [h]: 2025-04-07 08:00:00\n"
                "- Current TES energy [kWh]: 150.0\n"
                "\n"
                "Forecast data:\n"
                "Forecast features (raw values aligned with the RL look-ahead):\n"
                "\n"
                "- Remaining business hours today (08:00-17:00, close at 18:00):\n"
                "  - 08:00: load=19.3 kW, co2=0.395 kg-CO2/kWh  ←Now!!!\n"
                "  - 09:00: load=14.5 kW, co2=0.331 kg-CO2/kWh\n"
                "  - 10:00: load=14.8 kW, co2=0.270 kg-CO2/kWh\n"
                "  - 11:00: load=19.0 kW, co2=0.238 kg-CO2/kWh\n"
                "  - 12:00: load=14.1 kW, co2=0.259 kg-CO2/kWh\n"
                "  - 13:00: load=18.3 kW, co2=0.280 kg-CO2/kWh\n"
                "  - 14:00: load=17.0 kW, co2=0.326 kg-CO2/kWh\n"
                "  - 15:00: load=20.2 kW, co2=0.383 kg-CO2/kWh\n"
                "  - 16:00: load=17.0 kW, co2=0.428 kg-CO2/kWh\n"
                "  - 17:00: load=18.2 kW, co2=0.452 kg-CO2/kWh\n"
                "\n"
                "- Next-day planning metrics (for terminal planning):\n"
                "  - load_mean=17.3 kW (average cooling demand for next day)\n"
                "  - co2_low5_avg=0.247 kg-CO2/kWh (average of lowest 5 hours of CO2 factor for next day)\n"
                "  - co2_min=0.234 kg-CO2/kWh (minimum CO2 factor for next day)\n"
                "\n"
                "System parameters:\n"
                "ASHP rated capacity [kW]: 100.0\n"
                "ASHP base COP [-]: 4.0\n"
                "TES capacity [kWh]: 300.0\n"
                "\n"
                "Action space for the next hour:\n"
                "0 → ASHP output ratio = 0.00 (ASHP off; rely on TES if demand exists)\n"
                "1 → ASHP output ratio ≈ 0.33 (low output; TES covers most of the remaining demand)\n"
                "2 → ASHP output ratio ≈ 0.67 (medium output; TES supplements when load exceeds this level)\n"
                "3 → ASHP output ratio = 1.00 (full output; any surplus charges TES if capacity remains)\n"
                "\n"
                "Operational notes:\n"
                "- TES discharges automatically when load exceeds the scheduled ASHP output and energy is available.\n"
                "- TES charges automatically when ASHP output exceeds the load and free capacity exists.\n"
                "- Leaving demand unmet causes a large penalty; only select low ASHP ratios when TES has enough energy to bridge the gap.\n"
                "\n"
                "Decision requirements:\n"
                "- Optimise with a full-horizon perspective rather than a greedy step.\n"
                "- Keep TES utilisation efficient; avoid unnecessary saturation or depletion.\n"
                "- Prioritise emission reductions even if it requires near-term energy use.\n"
                "- Consider pre-charging during low-carbon periods and discharging during high-carbon periods while respecting TES energy limits.\n"
                "\n"
                "Return format:\n"
                "- Output a single token formatted as `[action_index]` (e.g., `[0]`, `[1]`, `[2]`, `[3]`)."
            ),
        },
    ]
    model_inputs = tokenizer.apply_chat_template(
        chat_prompt,
        add_generation_prompt=True,
        return_tensors="pt",
        pad_token_id=tokenizer.pad_token_id,
    )
    if isinstance(model_inputs, torch.Tensor):
        model_inputs = {"input_ids": model_inputs}
    else:
        model_inputs = {
            key: value
            for key, value in model_inputs.items()
        }
    if "attention_mask" not in model_inputs:
        model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"])
    model_inputs = {
        key: value.to(target_device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    print(f"--- Streaming response from {model_name} ({device}, 4-bit) ---\n", flush=True)

    with torch.inference_mode():
        model.generate(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=5000,
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
