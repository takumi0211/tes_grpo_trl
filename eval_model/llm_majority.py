"""Batch LLM querying utilities for majority-vote control."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from config import params  # noqa: F401 - re-exported for signature compatibility
from eval_model.llm import (
    _append_llm_log_row,
    _build_llm_messages,
    _format_messages_for_log,
    _GENERATION_KWARGS,
    _get_model,
    _get_tokenizer,
    _render_harmony_prompt,
    timeout_handler,
)


MAX_ATTEMPTS = 5
TIMEOUT_SECONDS = 200.0
ACTION_PATTERN = re.compile(r"\[\s*([0-3])\s*\]")
FALLBACK_DIGIT_PATTERN = re.compile(r"\b([0-3])\b")


@dataclass
class SampledAction:
    """Holds one sampled action from the LLM batch call."""

    sample_id: int
    thought: str
    action: int
    raw_response_text: str
    parsed_successfully: bool


def _call_hf_model_batch(messages: List[dict[str, str]], num_samples: int) -> list[str]:
    """Call the Hugging Face model once and decode ``num_samples`` completions."""
    tokenizer = _get_tokenizer()
    model = _get_model()
    harmony_prompt = _render_harmony_prompt(messages)
    inputs = tokenizer(harmony_prompt, return_tensors="pt").to("cuda")
    generation_kwargs = {
        **_GENERATION_KWARGS,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_return_sequences": num_samples,
    }
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    generated = outputs[:, prompt_len:]
    return tokenizer.batch_decode(generated, skip_special_tokens=False)


def _parse_action(text: str) -> tuple[int | None, bool]:
    """Extract an action index from model text."""
    match = ACTION_PATTERN.search(text)
    if match:
        return int(match.group(1)), True
    fallback = FALLBACK_DIGIT_PATTERN.search(text)
    if fallback:
        return int(fallback.group(1)), True
    return None, False


def get_llm_action_samples(
    current_time,
    df_text,
    current_storage,
    model_name,
    *,
    num_samples: int = 10,
    params=params,
) -> list[SampledAction]:
    """Generate ``num_samples`` candidate actions for a single prompt in parallel."""

    messages = _build_llm_messages(current_time, df_text, current_storage, params=params)
    prompt_text = _format_messages_for_log(messages)

    attempt = 0
    last_error = ""
    completions: list[str] = []

    while attempt < MAX_ATTEMPTS:
        attempt += 1
        try:
            with timeout_handler(int(TIMEOUT_SECONDS)):
                completions = _call_hf_model_batch(messages, num_samples)
            if completions:
                break
        except TimeoutError as exc:  # noqa: F821 - defined on non-Windows platforms
            last_error = f"LLM call timed out: {exc}"
            print(f"[get_llm_action_samples] {last_error} (attempt {attempt})")
        except Exception as exc:  # noqa: BLE001
            last_error = f"LLM request failed: {exc}"
            print(f"[get_llm_action_samples] {last_error} (attempt {attempt})")

    results: list[SampledAction] = []

    if not completions:
        fallback_reason = last_error or "Batch generation failed without an explicit error message."
        print("[get_llm_action_samples] Falling back to random actions for the entire batch.")
        for sample_id in range(1, num_samples + 1):
            fallback_action = int(np.random.choice([0, 1, 2, 3]))
            thought = (
                f"{fallback_reason} | randomly selected fallback action {fallback_action}."
            )
            sample = SampledAction(
                sample_id=sample_id,
                thought=thought,
                action=fallback_action,
                raw_response_text=thought,
                parsed_successfully=False,
            )
            results.append(sample)
            applied_action_json = json.dumps({"sample_id": sample_id, "action": fallback_action})
            _append_llm_log_row(
                prompt=prompt_text,
                applied_action_json=applied_action_json,
                output=thought,
            )
        return results

    for sample_id, raw_text in enumerate(completions, start=1):
        parsed_action, parsed_successfully = _parse_action(raw_text)
        if parsed_action is None:
            parsed_action = int(np.random.choice([0, 1, 2, 3]))
        thought = raw_text.strip() or "Model returned no textual content."
        sample = SampledAction(
            sample_id=sample_id,
            thought=thought,
            action=int(parsed_action),
            raw_response_text=raw_text,
            parsed_successfully=bool(parsed_successfully),
        )
        results.append(sample)

        applied_action_json = json.dumps({"sample_id": sample_id, "action": sample.action})
        _append_llm_log_row(
            prompt=prompt_text,
            applied_action_json=applied_action_json,
            output=raw_text,
        )

    return results
