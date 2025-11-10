"""
LLM制御モジュール
Hugging Face上のGRPOモデルにTES（蓄熱システム）の制御アクションを問い合わせる
"""
from __future__ import annotations

import csv
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import platform
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import params  # システムパラメータの読み込み

# Windows環境では signal.SIGALRM が使用できないため、条件分岐する
IS_WINDOWS = platform.system() == 'Windows'

if not IS_WINDOWS:
    import signal

@contextmanager
def timeout_handler(seconds):
    """
    LLM呼び出しにタイムアウトを設定するコンテキストマネージャー
    
    Args:
        seconds (int): タイムアウトまでの秒数
    
    Note:
        - Windows環境ではsignal.SIGALRMが使えないためタイムアウトは無効化される
        - Unix/Linux/macOS環境ではシグナルベースのタイムアウトが使用される
    """
    if IS_WINDOWS:
        # Windows環境ではタイムアウト機能を無効化
        yield
    else:
        # Unix/Linux/macOS環境ではsignalベースのタイムアウトを使用
        import signal
        def timeout_handler_fn(signum, frame):
            raise TimeoutError("LLM呼び出しがタイムアウトしました")
        old_handler = signal.signal(signal.SIGALRM, timeout_handler_fn)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


# Hugging Face上で使用するモデル設定
MODEL_ID = "takumi0211/tes_grpo"
MAX_NEW_TOKENS = 4000
TEMPERATURE = 0.8
TOP_P = 0.95
DO_SAMPLE = True

_GENERATION_KWARGS = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "do_sample": DO_SAMPLE,
}

_MODEL: AutoModelForCausalLM | None = None
_TOKENIZER: AutoTokenizer | None = None


# LLMとのやり取りをログに記録するためのカラム定義
LOG_COLUMNS = [
    "prompt",                  # LLMに送信したプロンプト全体
    "thought_process",         # LLMの思考プロセス（推論内容）
    "model_thinking",          # モデルの内部思考（APIが提供する場合）
    "applied_action_json",     # 実際に適用されたアクション（JSON形式）
    "raw_response_text",       # LLMからの生のレスポンステキスト
]
# ログファイルの保存先パス
LOG_PATH = Path(__file__).resolve().parent / "llm_logs.csv"


def _get_tokenizer() -> AutoTokenizer:
    """Hugging Faceトークナイザーを遅延ロードする"""
    global _TOKENIZER
    if _TOKENIZER is None:
        tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _TOKENIZER = tok
    return _TOKENIZER


def _get_model() -> AutoModelForCausalLM:
    """Hugging Faceモデルを遅延ロードする"""
    global _MODEL
    if _MODEL is None:
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA-capable GPU (H100) is required for MXFP4 inference.")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model.eval()
        _MODEL = model
    return _MODEL


def _render_harmony_prompt(messages: list[dict[str, str]]) -> str:
    """Harmony Chat形式でプロンプトを整形する"""
    tokenizer = _get_tokenizer()
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def _call_hf_model(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Harmony形式のメッセージをHugging Faceモデルに入力して生成する"""
    tokenizer = _get_tokenizer()
    model = _get_model()
    harmony_prompt = _render_harmony_prompt(messages)
    inputs = tokenizer(harmony_prompt, return_tensors="pt").to("cuda")
    generation_kwargs = {
        **_GENERATION_KWARGS,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    generated = outputs[:, prompt_len:]
    completion = tokenizer.batch_decode(generated, skip_special_tokens=False)[0]
    return {"output_text": completion}


def _safe_encode(text: Any) -> str:
    """
    テキストを安全にエンコードする（特殊なUnicode文字をASCII互換に変換）
    
    Args:
        text: エンコードするテキスト（任意の型）
    
    Returns:
        str: エンコードされた文字列
    
    Note:
        CSV保存時に問題を起こしやすい特殊文字（ダッシュ、引用符など）を
        ASCII互換の文字に置き換える
    """
    if text is None:
        return ""
    value = str(text)
    # 特殊なUnicode文字をASCII互換文字に変換
    replacements = {
        "\u2013": "-",      # en dash
        "\u2014": "--",     # em dash
        "\u2018": "'",      # left single quotation mark
        "\u2019": "'",      # right single quotation mark
        "\u201c": '"',      # left double quotation mark
        "\u201d": '"',      # right double quotation mark
        "\u2026": "...",    # horizontal ellipsis
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    return value


def _append_llm_log_row(*, prompt: str, thought_process: str, model_thinking: str, applied_action_json: str, raw_response_text: str) -> None:
    """
    LLMとのやり取りの1行をCSVログファイルに追記する
    
    Args:
        prompt: LLMに送信したプロンプト
        thought_process: LLMの思考プロセス
        model_thinking: モデルの内部思考
        applied_action_json: 適用されたアクション（JSON形式）
        raw_response_text: 生のレスポンステキスト
    
    Note:
        - ログディレクトリが存在しない場合は自動作成される
        - ファイルが存在しない場合はヘッダー行が追加される
        - UTF-8でのエンコードに失敗した場合はASCIIにフォールバックする
    """
    # ログディレクトリが存在しない場合は作成
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # ファイルが存在しないか空の場合はヘッダーが必要
    need_header = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
    # 各カラムのデータを安全にエンコード
    row = [
        _safe_encode(prompt),
        _safe_encode(thought_process),
        _safe_encode(model_thinking),
        _safe_encode(applied_action_json),
        _safe_encode(raw_response_text),
    ]
    try:
        # UTF-8でCSVに書き込み
        with LOG_PATH.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            if need_header:
                writer.writerow(LOG_COLUMNS)
            writer.writerow(row)
    except UnicodeEncodeError:
        # UTF-8での書き込みに失敗した場合はASCIIにフォールバック
        with LOG_PATH.open("a", newline="", encoding="ascii", errors="replace") as fp:
            writer = csv.writer(fp)
            if need_header:
                writer.writerow(LOG_COLUMNS)
            writer.writerow(row)


def _format_messages_for_log(messages: list[dict[str, str]]) -> str:
    """
    メッセージリストをログ用のテキスト形式に変換する
    
    Args:
        messages: メッセージの辞書のリスト（各辞書は'role'と'content'キーを持つ）
    
    Returns:
        str: "role: content"形式で改行区切りされたテキスト
    """
    return "\n".join(f"{msg.get('role', '')}: {msg.get('content', '')}" for msg in messages)


def _round_half_up(value: float) -> int:
    """
    四捨五入を行う（0.5の場合は切り上げ）
    
    Args:
        value: 丸める数値
    
    Returns:
        int: 丸められた整数値
    
    Note:
        Pythonの標準round()は銀行丸め（偶数丸め）を使用するため、
        一般的な四捨五入が必要な場合はこの関数を使用する
    """
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


# --- LLM に状態を渡してアクションを取得する関数 ---
def _build_system_prompt() -> str:
    """
    LLMのシステムプロンプトを構築する
    
    Returns:
        str: LLMの役割と目的を説明するシステムプロンプト
    
    Note:
        蓄熱システムの最適化エージェントとしての役割を定義し、
        CO2排出量の最小化が目的であることを明示する
    """
    return (
        "You are an optimisation agent that supervises a thermal energy storage (TES) plant. "
        "Your job is to minimise cumulative CO₂ emissions over the full simulation horizon by planning TES charge "
        "and discharge decisions."
    )


def _build_developer_prompt() -> str:
    """
    開発者向けの出力フォーマット指示プロンプトを構築する
    
    Returns:
        str: LLMの出力形式を指定するプロンプト
    
    Note:
        LLMが正しい形式（[0], [1], [2], [3]のいずれか）で
        アクションを返すように制約を設ける
    """
    return (
        "Developer instructions: respond using ASCII characters only. "
        "Return a single line formatted exactly as `[action_index]`, where action_index is an integer in {0, 1, 2, 3}. "
        "Do not include additional text, explanations, markdown, or keys."
    )


def _build_user_prompt(current_time, df_text, current_storage, params=params) -> str:
    """
    ユーザープロンプト（タスク説明と現在の状態情報）を構築する
    
    Args:
        current_time: 現在の時刻
        df_text: 予測データのテキスト（負荷、CO2係数など）
        current_storage: 現在のTES蓄熱量 [kWh]
        params: システムパラメータ辞書
    
    Returns:
        str: LLMに渡すユーザープロンプト
    
    Note:
        以下の情報を含む詳細なプロンプトを生成する：
        - 最適化目的（CO2排出量の最小化）
        - 現在の状態（時刻、蓄熱量）
        - 予測データ
        - システムパラメータ
        - アクション空間の説明（0〜3の4つの選択肢）
        - 運用上の注意事項
        - 意思決定の要件
    """
    # 蓄熱量を四捨五入して表示用に整形
    storage_display = _round_half_up(current_storage)
    # システムパラメータ情報を整形
    params_info = "\n".join(
        [
            f"ASHP rated capacity [kW]: {params['capacity_ashp']}",
            f"ASHP base COP [-]: {params['cop_ashp']}",
            f"TES capacity [kWh]: {params['capacity_tes']}",
        ]
    )
    return (
        f"Objective:\n"
        f"- Minimise total CO₂ emissions = electricity consumption × time-varying CO₂ intensity over the horizon.\n\n"
        f"Current context:\n"
        f"- Current time [h]: {current_time}\n"
        f"- Current TES energy [kWh]: {storage_display}\n\n"
        f"Forecast data:\n{df_text}\n\n"
        f"System parameters:\n{params_info}\n\n"
        "Action space for the next hour:\n"
        "0 → ASHP output ratio = 0.00 (ASHP off; rely on TES if demand exists)\n"
        "1 → ASHP output ratio ≈ 0.33 (low output; TES covers most of the remaining demand)\n"
        "2 → ASHP output ratio ≈ 0.67 (medium output; TES supplements when load exceeds this level)\n"
        "3 → ASHP output ratio = 1.00 (full output; any surplus charges TES if capacity remains)\n\n"
        "Operational notes:\n"
        "- TES discharges automatically when load exceeds the scheduled ASHP output and energy is available.\n"
        "- TES charges automatically when ASHP output exceeds the load and free capacity exists.\n"
        "- Always maintain TES capacity between minimum and maximum bounds (0 < capacity < 300); violating these constraints incurs a large penalty.\n\n"
        "Decision requirements:\n"
        "- Optimise with a full-horizon perspective rather than a greedy step.\n"
        "- Keep TES utilisation efficient; avoid unnecessary saturation or depletion.\n"
        "- Prioritise emission reductions even if it requires near-term energy use.\n"
        "- Consider pre-charging during low-carbon periods and discharging during high-carbon periods while respecting TES energy limits.\n"
        "- Consider next-day information when making decisions.\n\n"
        "Return format:\n"
        "- Output a single token formatted as `[action_index]` (e.g., `[0]`, `[1]`, `[2]`, `[3]`)."
    )


def _build_llm_messages(current_time, df_text, current_storage, params=params) -> list[dict[str, str]]:
    """
    LLMに送信するメッセージリストを構築する
    
    Args:
        current_time: 現在の時刻
        df_text: 予測データのテキスト
        current_storage: 現在のTES蓄熱量 [kWh]
        params: システムパラメータ辞書
    
    Returns:
        list[dict[str, str]]: メッセージリスト（systemとuserメッセージを含む）
    
    Note:
        2つのsystemメッセージ（役割定義と出力形式指示）と
        1つのuserメッセージ（タスク詳細）で構成される
    """
    return [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "system", "content": _build_developer_prompt()},
        {"role": "user", "content": _build_user_prompt(current_time, df_text, current_storage, params=params)},
    ]


def build_llm_prompt_text(current_time, df_text, current_storage, params=params) -> str:
    """
    LLMクエリ時に使用される完全なプロンプトテキストを返す
    
    Args:
        current_time: 現在の時刻
        df_text: 予測データのテキスト
        current_storage: 現在のTES蓄熱量 [kWh]
        params: システムパラメータ辞書
    
    Returns:
        str: フォーマットされたプロンプトテキスト
    
    Note:
        主にログ記録やデバッグ用に使用される
    """
    messages = _build_llm_messages(current_time, df_text, current_storage, params=params)
    return _format_messages_for_log(messages)


def get_llm_action(current_time, df_text, current_strage, model_name, params=params):
    """
    LLMにTES制御アクションを問い合わせる
    
    Args:
        current_time: 現在の時刻
        df_text: 予測データのテキスト
        current_strage: 現在のTES蓄熱量 [kWh]（変数名のtypoはそのまま維持）
        model_name: 互換性維持用の名目引数（実際にはMODEL_ID固定）
        params: システムパラメータ辞書
    
    Returns:
        tuple[str, int]: (思考プロセステキスト, アクションインデックス[0-3])
    
    Note:
        - 最大5回までリトライを行う
        - タイムアウトは200秒に設定
        - アクション抽出に失敗した場合はランダムアクションにフォールバック
        - 全てのやり取りはCSVログに記録される
    """

    def extract_output_and_thinking(response: Any) -> tuple[str, str]:
        """
        LLMレスポンスからテキストと思考プロセスを抽出する
        
        Args:
            response: LLMからのレスポンスオブジェクト
        
        Returns:
            tuple[str, str]: (出力テキスト, 思考プロセステキスト)
        
        Note:
            様々なレスポンス形式（Hugging Face推論、旧Ollama APIなど）に対応するため、
            複数のフィールド名を試行して情報を抽出する
        """

        def _get(obj: Any, name: str) -> Any:
            """オブジェクトから属性を取得（辞書とオブジェクトの両方に対応）"""
            if isinstance(obj, dict):
                return obj.get(name)
            return getattr(obj, name, None)

        def _collect_text(value: Any, bucket: list[str]) -> None:
            """
            値から表示用テキストを再帰的に収集する
            
            Note:
                文字列、リスト、辞書を再帰的に走査し、
                テキストコンテンツを抽出してbucketに追加する
            """
            if isinstance(value, str):
                text = value.strip()
                if text:
                    bucket.append(text)
            elif isinstance(value, list):
                for item in value:
                    _collect_text(item, bucket)
            elif isinstance(value, dict):
                # 一般的なテキストフィールド名を試行
                for candidate in ("text", "content", "output_text"):
                    if candidate in value:
                        _collect_text(value[candidate], bucket)

        def _collect_thinking(value: Any, bucket: list[str]) -> None:
            """
            値から思考プロセステキストを収集する
            
            Note:
                _collect_textと似ているが、思考プロセス専用
            """
            if isinstance(value, str):
                text = value.strip()
                if text:
                    bucket.append(text)

        def _dedupe(items: list[str]) -> list[str]:
            """
            重複を除去しながら順序を保持する
            
            Args:
                items: 文字列のリスト
            
            Returns:
                list[str]: 重複なしの文字列リスト（元の順序を保持）
            """
            seen: set[str] = set()
            ordered: list[str] = []
            for item in items:
                if item not in seen:
                    ordered.append(item)
                    seen.add(item)
            return ordered

        # テキストと思考プロセスを収集するためのバケット
        message_parts: list[str] = []
        thinking_parts: list[str] = []

        # 旧Ollama形式のレスポンスから抽出
        message_obj = _get(response, "message")
        if message_obj is not None:
            _collect_text(_get(message_obj, "content"), message_parts)
            _collect_thinking(_get(message_obj, "thinking"), thinking_parts)

        # トップレベルのoutput_textとthinkingフィールドから抽出
        _collect_text(_get(response, "output_text"), message_parts)
        _collect_thinking(_get(response, "thinking"), thinking_parts)

        # output配列形式のレスポンスから抽出
        output_items = _get(response, "output")
        if output_items:
            for item in output_items:
                if _get(item, "type") == "message":
                    _collect_text(_get(item, "content"), message_parts)
                elif _get(item, "type") == "reasoning":
                    _collect_thinking(_get(item, "content"), thinking_parts)

        # OpenAI形式のchoices配列から抽出
        choices = _get(response, "choices")
        if choices:
            for choice in choices:
                _collect_text(_get(choice, "text"), message_parts)
                message = _get(choice, "message")
                if message is not None:
                    _collect_text(_get(message, "content"), message_parts)
                    _collect_thinking(_get(message, "thinking"), thinking_parts)
                _collect_thinking(_get(choice, "reasoning"), thinking_parts)

        # 重複を除去
        message_parts = _dedupe(message_parts)
        thinking_parts = _dedupe(thinking_parts)
        # 改行で結合して返す
        return "\n".join(message_parts).strip(), "\n".join(thinking_parts).strip()

    # LLMに送信するメッセージリストを構築
    messages = _build_llm_messages(current_time, df_text, current_strage, params=params)
    prompt_text = _format_messages_for_log(messages)
    
    # リトライ設定
    max_attempts = 5  # 最大試行回数
    timeout_seconds = 200.0  # タイムアウト時間（秒）
    attempt = 0  # 現在の試行回数
    last_error = None  # 最後に発生したエラー
    last_raw_text = ""  # 最後に取得した生テキスト
    last_model_thinking = ""  # 最後に取得した思考プロセス

    # アクション抽出用の正規表現パターン（[0], [1], [2], [3]の形式を期待）
    action_pattern = re.compile(r"\[\s*([0-3])\s*\]")

    # リトライループ
    while attempt < max_attempts:
        attempt += 1
        try:
            # タイムアウト付きでLLMを呼び出し
            with timeout_handler(int(timeout_seconds)):
                response = _call_hf_model(messages)
        except TimeoutError as exc:
            # タイムアウトエラーの場合
            last_error = f"LLM呼び出しがタイムアウトしました: {exc}"
            print(f"[get_llm_action] {last_error} (attempt {attempt})")
            continue
        except Exception as exc:  # noqa: BLE001 - propagate readable info
            # その他のエラーの場合
            last_error = f"LLM request failed: {exc}"
            print(f"[get_llm_action] {last_error} (attempt {attempt})")
            continue

        # レスポンスからテキストと思考プロセスを抽出
        raw_text, model_thinking = extract_output_and_thinking(response)
        last_raw_text = raw_text
        last_model_thinking = model_thinking
        
        # テキストが空の場合はリトライ
        if not raw_text:
            last_error = "Model returned no textual content."
            print(f"[get_llm_action] {last_error} (attempt {attempt})")
            continue

        # アクションインデックスの抽出を試みる（主パターン: [0], [1], [2], [3]）
        match = action_pattern.search(raw_text)
        if not match:
            # フォールバック: 単独の数字（0-3）を探す
            digits_match = re.search(r"\b([0-3])\b", raw_text)
            if digits_match:
                match = digits_match

        # アクション抽出に失敗した場合はリトライ
        if not match:
            last_error = f"Could not parse action index from response: {raw_text}"
            print(f"[get_llm_action] {last_error} (attempt {attempt})")
            continue

        # 成功：アクションとthoughtを取得してログに記録
        action = int(match.group(1))
        thought = model_thinking.strip() if model_thinking else "Model did not expose reasoning via the API."
        if model_thinking:
            print(f"[get_llm_action] model thinking:\n{model_thinking}")
        print(raw_text)
        applied_action_json = json.dumps({"action": action})
        _append_llm_log_row(
            prompt=prompt_text,
            thought_process=thought,
            model_thinking=model_thinking,
            applied_action_json=applied_action_json,
            raw_response_text=raw_text,
        )
        return thought, action

    # 全ての試行が失敗した場合：フォールバック処理
    print("[get_llm_action] Falling back after repeated format errors.")
    print(prompt_text)
    if last_error:
        print(f"[get_llm_action] last error: {last_error}")
    
    # ランダムにアクションを選択
    fallback_action = int(np.random.choice([0, 1, 2, 3]))
    fallback_thought = last_model_thinking.strip() if last_model_thinking else "Error occurred during parsing; randomly selected an action after retries."
    
    # フォールバックアクションをログに記録
    applied_action_json = json.dumps({"action": fallback_action})
    raw_response_text = last_raw_text or (last_error or "")
    if not raw_response_text:
        raw_response_text = "No response captured."
    _append_llm_log_row(
        prompt=prompt_text,
        thought_process=fallback_thought,
        model_thinking=last_model_thinking,
        applied_action_json=applied_action_json,
        raw_response_text=raw_response_text,
    )
    return fallback_thought, fallback_action
