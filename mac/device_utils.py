# coding=utf-8
# Copyright 2026 Qwen3-TTS-Mac-GeneLab Contributors.
# SPDX-License-Identifier: Apache-2.0
"""
デバイス検出と dtype 自動選択

既知の MPS 制約:
- Voice Clone: float32 必須（float16 だと RuntimeError: probability tensor
  contains either inf, nan or element < 0）
- VoiceDesign: float16 OK
- CustomVoice: float16 OK
- FlashAttention 2: Mac 非対応 → sdpa を使用
- BFloat16: M1 では不安定な場合あり → float16 推奨
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from enum import Enum
from typing import Literal

import torch

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """TTS タスクの種類"""
    CUSTOM_VOICE = "custom_voice"
    VOICE_DESIGN = "voice_design"
    VOICE_CLONE = "voice_clone"


def is_apple_silicon() -> bool:
    """Apple Silicon Mac かどうかを判定する。

    Returns:
        bool: Apple Silicon Mac の場合 True
    """
    if platform.system() != "Darwin":
        return False

    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        cpu_brand = result.stdout.strip().lower()
        return "apple" in cpu_brand
    except (subprocess.SubprocessError, FileNotFoundError):
        # uname -m でフォールバック
        return platform.machine() == "arm64"


def is_mps_available() -> bool:
    """MPS (Metal Performance Shaders) が利用可能かどうかを判定する。

    Returns:
        bool: MPS が利用可能な場合 True
    """
    if not is_apple_silicon():
        return False

    try:
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()
    except AttributeError:
        return False


def get_optimal_device() -> str:
    """最適なデバイス文字列を返す。

    Returns:
        str: "mps", "cuda", または "cpu"
    """
    if is_mps_available():
        logger.info("MPS (Metal Performance Shaders) を使用します。")
        return "mps"
    elif torch.cuda.is_available():
        logger.info("CUDA を使用します。")
        return "cuda"
    else:
        logger.warning("GPU が利用できません。CPU を使用します。")
        return "cpu"


def get_optimal_dtype(
    task_type: TaskType | str,
    device: str | None = None,
) -> torch.dtype:
    """タスクに応じた最適な dtype を返す。

    Voice Clone は MPS で float32 が必須（float16 だとエラー）。
    他のタスクは float16 で高速化可能。

    Args:
        task_type: タスクの種類
        device: デバイス文字列（None の場合は自動検出）

    Returns:
        torch.dtype: 最適な dtype
    """
    if device is None:
        device = get_optimal_device()

    # 文字列を Enum に変換
    if isinstance(task_type, str):
        task_type = TaskType(task_type)

    # MPS の場合の dtype 選択
    if device == "mps":
        if task_type == TaskType.VOICE_CLONE:
            # Voice Clone は float32 必須
            logger.info("Voice Clone: MPS では float32 を使用します（既知の制約）。")
            return torch.float32
        else:
            # CustomVoice, VoiceDesign は float16 OK
            logger.info(f"{task_type.value}: float16 を使用します。")
            return torch.float16

    # CUDA の場合
    elif device == "cuda":
        # CUDA では bfloat16 が使える場合は使う
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    # CPU の場合
    else:
        return torch.float32


def get_attn_implementation() -> Literal["sdpa", "flash_attention_2", "eager"]:
    """最適な Attention 実装を返す。

    Mac では FlashAttention 2 が非対応のため、常に sdpa を返す。

    Returns:
        str: Attention 実装名
    """
    if is_apple_silicon():
        # Mac では FlashAttention 2 非対応
        logger.info("Apple Silicon: sdpa (Scaled Dot Product Attention) を使用します。")
        return "sdpa"

    # CUDA 環境では FlashAttention 2 を試す
    if torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            logger.info("FlashAttention 2 を使用します。")
            return "flash_attention_2"
        except ImportError:
            pass

    logger.info("sdpa を使用します。")
    return "sdpa"


def get_environment_vars() -> dict[str, str]:
    """必要な環境変数の dict を返す。

    Returns:
        dict[str, str]: 環境変数名と値のマッピング
    """
    env_vars = {
        # MPS のフォールバックを有効化
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        # MPS メモリ使用量の上限（0.0=無制限、PyTorch 2.10+ では 0.7 だと low watermark 超過エラー）
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
        # トークナイザーの並列処理を無効化（警告回避）
        "TOKENIZERS_PARALLELISM": "false",
    }

    return env_vars


def apply_environment_vars() -> None:
    """環境変数を現在のプロセスに適用する。"""
    env_vars = get_environment_vars()
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.debug(f"環境変数を設定: {key}={value}")


def get_mac_info() -> dict[str, str | int | bool]:
    """Mac のシステム情報を取得する。

    Returns:
        dict: システム情報
    """
    info: dict[str, str | int | bool] = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "is_apple_silicon": is_apple_silicon(),
        "mps_available": is_mps_available(),
    }

    # PyTorch 情報
    info["torch_version"] = torch.__version__

    # Apple Silicon の場合、チップ情報を取得
    if is_apple_silicon():
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            info["chip"] = result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            info["chip"] = "Unknown Apple Silicon"

        # メモリ情報
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            mem_bytes = int(result.stdout.strip())
            info["total_memory_gb"] = mem_bytes // (1024 ** 3)
        except (subprocess.SubprocessError, FileNotFoundError, ValueError):
            pass

    return info


# モジュールロード時に環境変数を適用
apply_environment_vars()
