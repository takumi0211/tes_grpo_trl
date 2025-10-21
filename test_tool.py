#!/usr/bin/env python3
"""
Agentic smoke test for unsloth/gpt-oss-20b using the official PythonTool.

The script mirrors test.py but follows the GPT-OSS documentation for enabling
the stateless python tool inside a harmony-formatted conversation. During the
dialogue the model can emit messages addressed to the `python` recipient; those
messages are executed via gpt_oss.tools.python_docker.docker_tool.PythonTool and
the outputs are appended back into the conversation before continuing.
"""

from __future__ import annotations

import asyncio
import datetime
import importlib.util
import os
import sys
import time
from typing import Sequence

import torch


def _configure_hf_download_stack() -> None:
    """Harden Hugging Face downloads against flaky Xet CAS outages."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _import_tooling():
    """
    Import GPT-OSS tooling with a user-friendly error if the dependencies are missing.
    Returns (PythonTool, harmony module).
    """
    try:
        from gpt_oss.tools.python_docker.docker_tool import PythonTool  # type: ignore
    except ImportError as exc:  # pragma: no cover - convenience message
        raise SystemExit(
            "The GPT-OSS PythonTool is unavailable. Install the `gpt-oss` package as described in "
            "the official README (https://github.com/openai/gpt-oss) before running this script."
        ) from exc

    try:
        from openai_harmony import (  # type: ignore
            Conversation,
            DeveloperContent,
            HarmonyEncodingName,
            Message,
            Role,
            SystemContent,
            load_harmony_encoding,
        )
    except ImportError as exc:  # pragma: no cover - convenience message
        raise SystemExit(
            "The `openai-harmony` package is required to render and parse GPT-OSS prompts. "
            "Install it (pip install openai-harmony) and retry."
        ) from exc

    return (
        PythonTool,
        Conversation,
        DeveloperContent,
        HarmonyEncodingName,
        Message,
        Role,
        SystemContent,
        load_harmony_encoding,
    )


def _extract_text(message) -> str:
    """Join all text fragments contained in a harmony message."""
    parts: list[str] = []
    for content in message.content:
        text = getattr(content, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


async def _run_python_tool(tool, message):
    """Execute a python tool call and collect all response messages."""
    responses = []
    async for response in tool.process(message):
        responses.append(response)
    return responses


async def _drive_conversation(
    *,
    model,
    conversation,
    python_tool,
    encoding,
    device: str,
    max_turns: int = 8,
) -> tuple[int, float]:
    total_generated = 0
    total_time = 0.0
    final_observed = False

    for turn in range(1, max_turns + 1):
        prompt_token_ids = encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )
        input_tensor = torch.tensor(
            [prompt_token_ids], device=device, dtype=torch.long
        )

        with torch.inference_mode():
            start = time.time()
            outputs = model.generate(
                input_ids=input_tensor,
                max_new_tokens=1024,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                return_dict_in_generate=True,
            )
            end = time.time()

        generated_sequence = outputs.sequences[0].tolist()
        new_tokens = generated_sequence[len(prompt_token_ids) :]
        total_generated += len(new_tokens)
        total_time += end - start

        assistant_messages = encoding.parse_messages_from_completion_tokens(
            new_tokens, Role.ASSISTANT
        )
        if not assistant_messages:
            print("No assistant messages were decoded; stopping.")
            break

        tool_invoked = False

        for message in assistant_messages:
            conversation.messages.append(message)
            author = message.author.role.value
            channel = message.channel or "default"
            text = _extract_text(message)
            print(f"[assistant:{author}:{channel}] {text}\n")

            if message.recipient == python_tool.name:
                responses = await _run_python_tool(python_tool, message)
                if not responses:
                    print("[python] Tool produced no output.\n")
                for response in responses:
                    conversation.messages.append(response)
                    print(f"[python:tool] {_extract_text(response)}\n")
                tool_invoked = True
                break

            if (
                message.author.role == Role.ASSISTANT
                and (message.channel == "final" or message.recipient is None)
            ):
                if text.strip().startswith("[") and text.strip().endswith("]"):
                    final_observed = True
                    break

        if final_observed:
            break

        if tool_invoked:
            continue

        # If we reach here the assistant did not emit a tool call or final answer.
        print("Assistant response was inconclusive; stopping.")
        break

    return total_generated, total_time


async def _async_main() -> None:
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

    (
        PythonTool,
        Conversation,
        DeveloperContent,
        HarmonyEncodingName,
        Message,
        Role,
        SystemContent,
        load_harmony_encoding,
    ) = _import_tooling()

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

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    python_tool = PythonTool()

    system_content = (
        SystemContent.new()
        .with_model_identity(
            "You are an optimisation agent that supervises a thermal energy storage (TES) plant. "
            "Your job is to minimise cumulative CO2 emissions over the full simulation horizon by planning TES charge and discharge decisions."
        )
        .with_conversation_start_date(datetime.date.today().isoformat())
        .with_tools(python_tool.tool_config)
        .with_required_channels(["analysis", "commentary", "final"])
    )

    developer_message = Message.from_role_and_content(
        Role.DEVELOPER,
        DeveloperContent.new().with_instructions(
            "Respond using ASCII characters only. When you are ready to answer the decision task, "
            "return a single line formatted exactly as `[action_index]` with action_index in {0,1,2,3}. "
            "Use the python tool for quantitative reasoning when helpful and do not expose hidden reasoning to the user."
        ),
    )

    user_prompt = (
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
    )

    system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
    user_message = Message.from_role_and_content(Role.USER, user_prompt)

    conversation = Conversation.from_messages(
        [system_message, developer_message, user_message]
    )

    print(f"--- Running {model_name} with PythonTool support ---\n")
    try:
        generated, elapsed = await _drive_conversation(
            model=model,
            conversation=conversation,
            python_tool=python_tool,
            encoding=encoding,
            device=device,
        )
    finally:
        python_tool.close()

    tokens_per_second = generated / elapsed if elapsed > 0 else 0.0
    print(
        f"Generated {generated} tokens across {elapsed:.2f} seconds "
        f"({tokens_per_second:.2f} tokens/s)"
    )


def main(argv: Sequence[str] | None = None) -> None:  # noqa: ARG001 - parity with test.py signature
    asyncio.run(_async_main())


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        sys.stderr.write(str(exc) + "\n")
        sys.exit(exc.code)
