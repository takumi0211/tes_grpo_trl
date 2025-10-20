#!/usr/bin/env python
"""
Quick smoke test for unsloth/gpt-oss-20b 4-bit inference with streaming output.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch
import time


def _configure_hf_download_stack() -> None:
    """Harden Hugging Face downloads against flaky Xet CAS outages."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA device is required to run unsloth/gpt-oss-20b.")

    _configure_hf_download_stack()

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

    device = "cuda"
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
        {
            "role": "system",
            "content": (
                "You are an optimisation agent that supervises a thermal energy storage (TES) plant. "
                "Your job is to minimise cumulative CO2 emissions over the full simulation horizon by planning "
                "TES charge and discharge decisions."
            ),
        },
        {
            "role": "system",
            "content": (
                "Developer instructions: respond using ASCII characters only. "
                "Return a single line formatted exactly as `[action_index]`, where action_index is an integer in {0, 1, 2, 3}. "
                "Do not include additional text, explanations, markdown, or keys."
            ),
        },
        {
            "role": "user",
            "content": (
                "Objective:\n"
                "- Minimise total CO2 emissions = electricity consumption x time-varying CO2 intensity over the horizon.\n\n"
                "Current context:\n"
                "- Current time [h]: 2025-04-07 08:00:00\n"
                "- Current TES energy [kWh]: 150.0\n\n"
                "Forecast data:\n"
                "Forecast features (raw values aligned with the RL look-ahead):\n\n"
                "- Remaining business hours today (08:00-17:00, close at 18:00):\n"
                "  - 08:00: load=19.3 kW, co2=0.395 kg-CO2/kWh  <- Now!!!\n"
                "  - 09:00: load=14.5 kW, co2=0.331 kg-CO2/kWh\n"
                "  - 10:00: load=14.8 kW, co2=0.270 kg-CO2/kWh\n"
                "  - 11:00: load=19.0 kW, co2=0.238 kg-CO2/kWh\n"
                "  - 12:00: load=14.1 kW, co2=0.259 kg-CO2/kWh\n"
                "  - 13:00: load=18.3 kW, co2=0.280 kg-CO2/kWh\n"
                "  - 14:00: load=17.0 kW, co2=0.326 kg-CO2/kWh\n"
                "  - 15:00: load=20.2 kW, co2=0.383 kg-CO2/kWh\n"
                "  - 16:00: load=17.0 kW, co2=0.428 kg-CO2/kWh\n"
                "  - 17:00: load=18.2 kW, co2=0.452 kg-CO2/kWh\n\n"
                "- Next-day planning metrics (for terminal planning):\n"
                "  - load_mean=17.3 kW (average cooling demand for next day)\n"
                "  - co2_low5_avg=0.247 kg-CO2/kWh (average of lowest 5 hours of CO2 factor for next day)\n"
                "  - co2_min=0.234 kg-CO2/kWh (minimum CO2 factor for next day)\n\n"
                "System parameters:\n"
                "ASHP rated capacity [kW]: 100.0\n"
                "ASHP base COP [-]: 4.0\n"
                "TES capacity [kWh]: 300.0\n\n"
                "Action space for the next hour:\n"
                "0 -> ASHP output ratio = 0.00 (ASHP off; rely on TES if demand exists)\n"
                "1 -> ASHP output ratio ~= 0.33 (low output; TES covers most of the remaining demand)\n"
                "2 -> ASHP output ratio ~= 0.67 (medium output; TES supplements when load exceeds this level)\n"
                "3 -> ASHP output ratio = 1.00 (full output; any surplus charges TES if capacity remains)\n\n"
                "Operational notes:\n"
                "- TES discharges automatically when load exceeds the scheduled ASHP output and energy is available.\n"
                "- TES charges automatically when ASHP output exceeds the load and free capacity exists.\n"
                "- Leaving demand unmet causes a large penalty; only select low ASHP ratios when TES has enough energy to bridge the gap.\n\n"
                "Decision requirements:\n"
                "- Optimise with a full-horizon perspective rather than a greedy step.\n"
                "- Keep TES utilisation efficient; avoid unnecessary saturation or depletion.\n"
                "- Prioritise emission reductions even if it requires near-term energy use.\n"
                "- Consider pre-charging during low-carbon periods and discharging during high-carbon periods while respecting TES energy limits.\n\n"
                "Return format:\n"
                "- Output a single token formatted as `[action_index]` (e.g., `[0]`, `[1]`, `[2]`, `[3]`)."
            ),
        },
    ]
    model_inputs = tokenizer.apply_chat_template(
        chat_prompt,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(model_inputs, torch.Tensor):
        model_inputs = {"input_ids": model_inputs.to(device)}
    else:
        if hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(device)
        model_inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    print(f"--- Streaming response from {model_name} ({device}, 4-bit) ---\n", flush=True)

    with torch.inference_mode():
        start_time = time.time()
        outputs = model.generate(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=5000,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            return_dict_in_generate=True,
        )
        end_time = time.time()
        input_length = len(model_inputs["input_ids"][0])
        generated_tokens = len(outputs.sequences[0]) - input_length
        time_taken = end_time - start_time
        tokens_per_second = generated_tokens / time_taken if time_taken > 0 else 0
        print(f"\nGenerated {generated_tokens} tokens in {time_taken:.2f} seconds ({tokens_per_second:.2f} tokens/s)")
    print("\n--- Done ---")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        sys.stderr.write(str(exc) + "\n")
        sys.exit(exc.code)
