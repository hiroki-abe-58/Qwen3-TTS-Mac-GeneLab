# coding=utf-8
# Copyright 2026 Qwen3-TTS-Mac-GeneLab Contributors.
# SPDX-License-Identifier: Apache-2.0
"""
カスタムボイスタブ

9種のプリセットスピーカーを選択し、感情指示と言語を指定して音声を生成する。
"""

from __future__ import annotations

import logging
import re
import tempfile
import time
from typing import Any

import gradio as gr
import numpy as np

from ui.i18n_utils import t

logger = logging.getLogger(__name__)

# スピーカー名一覧（MLX 8bit モデル準拠。表示名・説明は i18n から取得）
SPEAKER_KEYS = [
    "serena", "vivian", "aiden", "ryan", "eric",
    "dylan", "ono_anna", "sohee", "uncle_fu",
]

# TTS エンジン用言語キー（値は固定英語キー）
LANGUAGE_KEYS = [
    "Japanese", "English", "Chinese", "Korean",
    "French", "German", "Spanish", "Italian",
    "Portuguese", "Russian",
]

# 感情キー
EMOTION_KEYS = [
    "neutral", "happy", "sad", "angry",
    "surprised", "fearful", "disgusted",
    "calm", "excited", "tender",
]


def _speaker_choices() -> list[tuple[str, str]]:
    """i18n を使ったスピーカー選択肢 (表示名, キー) を生成する。"""
    choices: list[tuple[str, str]] = []
    for key in SPEAKER_KEYS:
        local = t(f"speakers.{key}.local_name", key)
        desc = t(f"speakers.{key}.description", "")
        label = f"{local} - {desc}" if desc else local
        choices.append((label, key))
    return choices


def _language_choices() -> list[tuple[str, str]]:
    """i18n を使った言語選択肢（表示名, エンジンキー）を生成する。"""
    return [(t(f"languages.{k}", k), k) for k in LANGUAGE_KEYS]


def _emotion_choices() -> list[tuple[str, str]]:
    """i18n を使った感情選択肢を生成する。"""
    return [(t(f"emotions.{k}", k), k) for k in EMOTION_KEYS]


def extract_speaker_name(selection: str) -> str:
    """選択値からスピーカー名を返す。Dropdown の tuple 形式ではキーがそのまま渡される。"""
    if selection in SPEAKER_KEYS:
        return selection
    match = re.search(r"（([\w_]+)）", selection)
    return match.group(1) if match else "serena"


def generate_audio(
    text: str,
    speaker: str,
    language: str,
    emotion: str,
    speed: float,
) -> tuple[Any, str]:
    """音声を生成する。"""
    if not text.strip():
        return None, t("messages.enter_text")

    try:
        from mac.engine import DualEngine, TaskType

        logger.info(f"音声生成開始: speaker={speaker}, language={language}, emotion={emotion}")
        start_time = time.time()

        engine = DualEngine()
        result = engine.generate(
            text=text,
            task_type=TaskType.CUSTOM_VOICE,
            language=language,
            speaker=speaker,
            emotion=emotion if emotion != "neutral" else None,
            speed=speed,
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import soundfile as sf
            sf.write(f.name, result.audio, result.sample_rate)
            audio_path = f.name

        elapsed = time.time() - start_time
        status = (
            f"{t('messages.generated')}: {result.duration_seconds:.2f}s | "
            f"{elapsed:.2f}s | {result.engine_used.value}"
        )
        logger.info(status)
        return audio_path, status

    except Exception as e:
        logger.error(f"音声生成エラー: {e}")
        return None, f"{t('messages.error')}: {str(e)}"


def create_custom_voice_tab() -> None:
    """カスタムボイスタブを作成する。"""
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label=t("custom_voice.text_input.label"),
                placeholder=t("custom_voice.text_input.placeholder"),
                lines=5,
                max_lines=10,
            )

            speaker_selector = gr.Dropdown(
                choices=_speaker_choices(),
                value=SPEAKER_KEYS[0],
                label=t("custom_voice.speaker_selector.label"),
                info=t("custom_voice.speaker_selector.info"),
                interactive=True,
            )

            with gr.Row():
                language_selector = gr.Dropdown(
                    choices=_language_choices(),
                    value="Japanese",
                    label=t("custom_voice.language_selector.label"),
                    info=t("custom_voice.language_selector.info"),
                    interactive=True,
                )

                emotion_selector = gr.Dropdown(
                    choices=_emotion_choices(),
                    value="neutral",
                    label=t("custom_voice.emotion_selector.label"),
                    info=t("custom_voice.emotion_selector.info"),
                    interactive=True,
                )

            speed_slider = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label=t("custom_voice.speed_slider.label"),
                info=t("custom_voice.speed_slider.info"),
                interactive=True,
            )

            generate_btn = gr.Button(
                t("custom_voice.generate_button"),
                variant="primary",
                elem_classes=["primary-btn"],
            )

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label=t("custom_voice.audio_output"),
                type="filepath",
                interactive=False,
            )

            status_output = gr.Textbox(
                label=t("custom_voice.status"),
                interactive=False,
                lines=2,
            )

            gr.Examples(
                examples=[
                    ["Hello, welcome to Qwen3-TTS. This is a demonstration of voice synthesis."],
                    ["The quick brown fox jumps over the lazy dog."],
                    ["Today's weather is sunny with a gentle breeze."],
                ],
                inputs=[text_input],
                label=t("custom_voice.sample_texts"),
            )

    def on_generate(text: str, speaker_sel: str, language: str, emotion: str, speed: float) -> tuple:
        speaker = extract_speaker_name(speaker_sel)
        return generate_audio(text, speaker, language, emotion, speed)

    generate_btn.click(
        fn=on_generate,
        inputs=[text_input, speaker_selector, language_selector, emotion_selector, speed_slider],
        outputs=[audio_output, status_output],
    )
