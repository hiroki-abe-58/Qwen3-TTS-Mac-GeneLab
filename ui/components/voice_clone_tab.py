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

logger = logging.getLogger(__name__)

# 言語選択
LANGUAGE_CHOICES = [
    "Japanese", "English", "Chinese", "Korean",
    "French", "German", "Spanish", "Italian",
    "Portuguese", "Russian",
]


def transcribe_audio(audio_path: str, language: str) -> str:
    """音声を書き起こす。

    Args:
        audio_path: 音声ファイルパス
        language: 言語

    Returns:
        書き起こしテキスト
    """
    if not audio_path:
        return ""

    try:
        # Whisper transcriber を使用
        from mac.whisper_transcriber import WhisperTranscriber

        transcriber = WhisperTranscriber()
        result = transcriber.transcribe(audio_path, language=language)
        logger.info(f"書き起こし完了: {result[:50]}...")
        return result

    except ImportError:
        logger.warning("WhisperTranscriber が利用できません。手動入力が必要です。")
        return "[Whisper が利用できません。手動で入力してください]"
    except Exception as e:
        logger.error(f"書き起こしエラー: {e}")
        return f"[書き起こしエラー: {str(e)}]"


def generate_cloned_audio(
    text: str,
    reference_audio: str,
    reference_text: str,
    language: str,
    speed: float,
) -> tuple[Any, str]:
    """ボイスクローンで音声を生成する。

    Args:
        text: 読み上げテキスト
        reference_audio: 参照音声パス
        reference_text: 参照音声のテキスト
        language: 言語
        speed: 速度

    Returns:
        tuple: (音声ファイルパス, ステータスメッセージ)
    """
    if not text.strip():
        return None, "読み上げテキストを入力してください。"

    if not reference_audio:
        return None, "参照音声をアップロードしてください。"

    if not reference_text.strip():
        return None, "参照音声のテキストを入力してください（Whisper で自動書き起こし可能）。"

    try:
        from mac.engine import DualEngine, TaskType
        import librosa

        logger.info("ボイスクローン生成開始...")
        start_time = time.time()

        # 参照音声を読み込み
        ref_audio, ref_sr = librosa.load(reference_audio, sr=None)
        logger.info(f"参照音声: {len(ref_audio)/ref_sr:.2f}秒, {ref_sr}Hz")

        # エンジン取得
        engine = DualEngine()

        # 生成
        result = engine.generate(
            text=text,
            task_type=TaskType.VOICE_CLONE,
            language=language,
            reference_audio=ref_audio,
            reference_text=reference_text,
            reference_sr=ref_sr,
            speed=speed,
        )

        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import soundfile as sf
            sf.write(f.name, result.audio, result.sample_rate)
            audio_path = f.name

        elapsed = time.time() - start_time
        status = (
            f"生成完了: {result.duration_seconds:.2f}秒の音声 | "
            f"処理時間: {elapsed:.2f}秒 | "
            f"エンジン: {result.engine_used.value}"
        )

        logger.info(status)
        return audio_path, status

    except Exception as e:
        logger.error(f"ボイスクローン生成エラー: {e}")
        return None, f"エラー: {str(e)}"


def create_voice_clone_tab() -> None:
    """ボイスクローンタブを作成する。"""
    gr.Markdown(
        """
        ### ボイスクローン
        
        わずか3秒の参照音声から、その声で新しいテキストを読み上げます。
        参照音声は明瞭で、背景ノイズが少ないものを使用してください。
        
        > **注意**: Voice Clone は PyTorch MPS エンジンで float32 を使用します（MLX より多くのメモリが必要）。
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            # 参照音声
            reference_audio = gr.Audio(
                label="参照音声（3〜30秒推奨）",
                type="filepath",
                sources=["upload", "microphone"],
            )

            # 書き起こしボタンと言語選択
            with gr.Row():
                transcribe_language = gr.Dropdown(
                    choices=LANGUAGE_CHOICES,
                    value="Japanese",
                    label="参照音声の言語",
                    scale=2,
                )
                transcribe_btn = gr.Button(
                    "Whisper で書き起こし",
                    variant="secondary",
                    scale=1,
                )

            # 参照音声のテキスト
            reference_text = gr.Textbox(
                label="参照音声のテキスト",
                placeholder="参照音声で話されている内容を入力、または上のボタンで自動書き起こし...",
                lines=3,
                info="正確なテキストを入力すると、より高品質なクローンが可能です",
            )

            gr.Markdown("---")

            # 生成テキスト
            text_input = gr.Textbox(
                label="読み上げテキスト",
                placeholder="クローンした声で読み上げたいテキストを入力してください...",
                lines=5,
                max_lines=10,
            )

            with gr.Row():
                # 出力言語選択
                output_language = gr.Dropdown(
                    choices=LANGUAGE_CHOICES,
                    value="Japanese",
                    label="出力言語",
                    info="生成音声の言語",
                )

                # 速度調整
                speed_slider = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="速度",
                    info="0.5（遅い）〜 2.0（速い）",
                )

            # 生成ボタン
            generate_btn = gr.Button(
                "音声を生成",
                variant="primary",
                elem_classes=["primary-btn"],
            )

        with gr.Column(scale=1):
            # 出力
            audio_output = gr.Audio(
                label="生成された音声",
                type="filepath",
                interactive=False,
            )

            # ステータス
            status_output = gr.Textbox(
                label="ステータス",
                interactive=False,
                lines=2,
            )

            # ヒント
            gr.Markdown(
                """
                #### ヒント
                
                - 参照音声は **3〜30秒** が最適
                - **静かな環境** で録音された音声を使用
                - 参照音声のテキストは **正確に** 入力
                - 多言語対応: 日本語の声で英語を話すことも可能
                """
            )

    # サンプルテキスト
    gr.Examples(
        examples=[
            ["こんにちは、音声クローン技術のデモンストレーションです。"],
            ["本日は晴天なり。マイクテスト、マイクテスト。"],
            ["Hello, this is a voice cloning demonstration."],
        ],
        inputs=[text_input],
        label="サンプルテキスト",
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
