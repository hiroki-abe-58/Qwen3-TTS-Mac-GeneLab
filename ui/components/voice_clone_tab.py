# coding=utf-8
# Copyright 2026 Qwen3-TTS-Mac-GeneLab Contributors.
# SPDX-License-Identifier: Apache-2.0
"""
ボイスクローンタブ

参照音声をアップロードし、その声で新しいテキストを読み上げる。
Whisper による自動書き起こし機能を搭載。
"""

from __future__ import annotations

import logging
import tempfile
import time
from typing import Any

import gradio as gr
import numpy as np

from ui.i18n_utils import t, t_list

logger = logging.getLogger(__name__)

# TTS エンジン用言語キー
LANGUAGE_KEYS = [
    "Japanese", "English", "Chinese", "Korean",
    "French", "German", "Spanish", "Italian",
    "Portuguese", "Russian",
]


def _language_choices() -> list[tuple[str, str]]:
    return [(t(f"languages.{k}", k), k) for k in LANGUAGE_KEYS]


def transcribe_audio(audio_path: str, language: str) -> str:
    """音声を書き起こす。"""
    if not audio_path:
        return ""

    try:
        from mac.whisper_transcriber import WhisperTranscriber

        transcriber = WhisperTranscriber()
        result = transcriber.transcribe(audio_path, language=language)
        logger.info(f"書き起こし完了: {result[:50]}...")
        return result

    except ImportError:
        logger.warning("WhisperTranscriber が利用できません。")
        return "[Whisper unavailable]"
    except Exception as e:
        logger.error(f"書き起こしエラー: {e}")
        return f"[Error: {str(e)}]"


def generate_cloned_audio(
    text: str,
    reference_audio: str,
    reference_text: str,
    language: str,
    speed: float,
) -> tuple[Any, str]:
    """ボイスクローンで音声を生成する。"""
    if not text.strip():
        return None, t("messages.enter_text")

    if not reference_audio:
        return None, t("messages.upload_reference")

    if not reference_text.strip():
        return None, t("messages.enter_reference_text")

    try:
        from mac.engine import DualEngine, TaskType
        import librosa

        logger.info("ボイスクローン生成開始...")
        start_time = time.time()

        ref_audio, ref_sr = librosa.load(reference_audio, sr=None)
        logger.info(f"参照音声: {len(ref_audio)/ref_sr:.2f}秒, {ref_sr}Hz")

        engine = DualEngine()
        result = engine.generate(
            text=text,
            task_type=TaskType.VOICE_CLONE,
            language=language,
            reference_audio=ref_audio,
            reference_text=reference_text,
            reference_sr=ref_sr,
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
        logger.error(f"ボイスクローン生成エラー: {e}")
        return None, f"{t('messages.error')}: {str(e)}"


def create_voice_clone_tab() -> None:
    """ボイスクローンタブを作成する。"""
    gr.Markdown(
        f"### {t('voice_clone.title')}\n\n"
        f"{t('voice_clone.description')}\n\n"
        f"> {t('voice_clone.notice')}"
    )

    with gr.Row():
        with gr.Column(scale=2):
            reference_audio = gr.Audio(
                label=t("voice_clone.reference_audio.label"),
                type="filepath",
                sources=["upload", "microphone"],
            )

            with gr.Row():
                transcribe_language = gr.Dropdown(
                    choices=_language_choices(),
                    value="Japanese",
                    label=t("voice_clone.reference_language.label"),
                    scale=2,
                    interactive=True,
                )
                transcribe_btn = gr.Button(
                    t("voice_clone.transcribe_button"),
                    variant="secondary",
                    scale=1,
                )

            reference_text = gr.Textbox(
                label=t("voice_clone.reference_text.label"),
                placeholder=t("voice_clone.reference_text.placeholder"),
                lines=3,
                info=t("voice_clone.reference_text.info"),
            )

            gr.Markdown("---")

            text_input = gr.Textbox(
                label=t("voice_clone.text_input.label"),
                placeholder=t("voice_clone.text_input.placeholder"),
                lines=5,
                max_lines=10,
            )

            with gr.Row():
                output_language = gr.Dropdown(
                    choices=_language_choices(),
                    value="Japanese",
                    label=t("voice_clone.output_language.label"),
                    info=t("voice_clone.output_language.info"),
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
                t("voice_clone.generate_button"),
                variant="primary",
                elem_classes=["primary-btn"],
            )

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label=t("voice_clone.audio_output"),
                type="filepath",
                interactive=False,
            )

            status_output = gr.Textbox(
                label=t("voice_clone.status"),
                interactive=False,
                lines=2,
            )

            tips_list = t_list("voice_clone.tips.items")
            tips_items = "\n".join(f"- {item}" for item in tips_list if item)
            gr.Markdown(f"#### {t('voice_clone.tips.title')}\n\n{tips_items}")

    gr.Examples(
        examples=[
            ["Hello, this is a voice cloning demonstration."],
            ["The weather today is sunny and warm."],
        ],
        inputs=[text_input],
        label=t("voice_clone.sample_texts"),
    )

    # イベントハンドラ
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[reference_audio, transcribe_language],
        outputs=[reference_text],
    )

    generate_btn.click(
        fn=generate_cloned_audio,
        inputs=[text_input, reference_audio, reference_text, output_language, speed_slider],
        outputs=[audio_output, status_output],
    )
