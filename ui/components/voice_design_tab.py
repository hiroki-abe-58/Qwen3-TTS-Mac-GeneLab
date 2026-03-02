# coding=utf-8
# Copyright 2026 Qwen3-TTS-Mac-GeneLab Contributors.
# SPDX-License-Identifier: Apache-2.0
"""
ボイスデザインタブ

テキスト記述でボイスの特徴を指定し、音声を生成する。
"""

from __future__ import annotations

import logging
import tempfile
import time
from typing import Any

import gradio as gr

from ui.i18n_utils import t

logger = logging.getLogger(__name__)

# TTS エンジン用言語キー
LANGUAGE_KEYS = [
    "Japanese", "English", "Chinese", "Korean",
    "French", "German", "Spanish", "Italian",
    "Portuguese", "Russian",
]

# ボイスデザインのサンプル（説明は英語固定 = エンジン入力）
VOICE_DESIGN_SAMPLES = [
    ("Calm middle-aged male", "A calm and composed middle-aged male voice with a warm, reassuring tone."),
    ("Energetic young female", "An energetic young female voice, cheerful and lively with a bright tone."),
    ("Professional narrator", "A professional narrator voice, clear and articulate, suitable for documentaries."),
    ("Gentle grandmother", "A gentle elderly female voice, warm and comforting like a grandmother."),
    ("Powerful announcer", "A powerful male announcer voice, confident and authoritative."),
    ("Cute young girl", "A cute young girl voice, sweet and innocent with a playful tone."),
]


def _language_choices() -> list[tuple[str, str]]:
    return [(t(f"languages.{k}", k), k) for k in LANGUAGE_KEYS]


def generate_audio_with_design(
    text: str,
    voice_description: str,
    language: str,
    speed: float,
) -> tuple[Any, str]:
    """ボイスデザインで音声を生成する。"""
    if not text.strip():
        return None, t("messages.enter_text")

    if not voice_description.strip():
        return None, t("messages.enter_voice_description")

    try:
        from mac.engine import DualEngine, TaskType

        logger.info(f"ボイスデザイン生成開始: description={voice_description[:50]}...")
        start_time = time.time()

        engine = DualEngine()
        result = engine.generate(
            text=text,
            task_type=TaskType.VOICE_DESIGN,
            language=language,
            voice_description=voice_description,
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
        logger.error(f"ボイスデザイン生成エラー: {e}")
        return None, f"{t('messages.error')}: {str(e)}"


def create_voice_design_tab() -> None:
    """ボイスデザインタブを作成する。"""
    gr.Markdown(f"### {t('voice_design.title')}\n\n{t('voice_design.description')}")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label=t("voice_design.text_input.label"),
                placeholder=t("voice_design.text_input.placeholder"),
                lines=5,
                max_lines=10,
            )

            voice_description = gr.Textbox(
                label=t("voice_design.voice_description.label"),
                placeholder=t("voice_design.voice_description.placeholder"),
                lines=3,
                info=t("voice_design.voice_description.info"),
            )

            with gr.Row():
                language_selector = gr.Dropdown(
                    choices=_language_choices(),
                    value="Japanese",
                    label=t("custom_voice.language_selector.label"),
                    info=t("custom_voice.language_selector.info"),
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
                t("voice_design.generate_button"),
                variant="primary",
                elem_classes=["primary-btn"],
            )

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label=t("voice_design.audio_output"),
                type="filepath",
                interactive=False,
            )

            status_output = gr.Textbox(
                label=t("voice_design.status"),
                interactive=False,
                lines=2,
            )

    # サンプル
    gr.Markdown(f"### {t('voice_design.samples_title')}")
    with gr.Row():
        for label, desc_en in VOICE_DESIGN_SAMPLES[:3]:
            with gr.Column():
                gr.Markdown(f"**{label}**")
                sample_btn = gr.Button(t("voice_design.use_button"), size="sm")

                def make_handler(d: str):
                    return lambda: d
                sample_btn.click(fn=make_handler(desc_en), outputs=[voice_description])

    with gr.Row():
        for label, desc_en in VOICE_DESIGN_SAMPLES[3:]:
            with gr.Column():
                gr.Markdown(f"**{label}**")
                sample_btn = gr.Button(t("voice_design.use_button"), size="sm")

                def make_handler(d: str):
                    return lambda: d
                sample_btn.click(fn=make_handler(desc_en), outputs=[voice_description])

    # イベントハンドラ
    generate_btn.click(
        fn=generate_audio_with_design,
        inputs=[text_input, voice_description, language_selector, speed_slider],
        outputs=[audio_output, status_output],
    )
