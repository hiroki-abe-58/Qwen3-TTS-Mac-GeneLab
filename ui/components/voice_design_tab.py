# coding=utf-8
# Copyright 2026 Qwen3-TTS-Mac-GeneLab Contributors.
# SPDX-License-Identifier: Apache-2.0
"""
ボイスデザインタブ

テキスト記述でボイスの特徴を指定し、音声を生成する。
例: 「落ち着いた中年男性の声」「元気な若い女性の声」
"""

from __future__ import annotations

import logging
import tempfile
import time
from typing import Any

import gradio as gr

logger = logging.getLogger(__name__)

# 言語選択
LANGUAGE_CHOICES = [
    "Japanese", "English", "Chinese", "Korean",
    "French", "German", "Spanish", "Italian",
    "Portuguese", "Russian",
]

# ボイスデザインのサンプル
VOICE_DESIGN_SAMPLES = [
    ("落ち着いた中年男性", "A calm and composed middle-aged male voice with a warm, reassuring tone."),
    ("元気な若い女性", "An energetic young female voice, cheerful and lively with a bright tone."),
    ("知的なナレーター", "A professional narrator voice, clear and articulate, suitable for documentaries."),
    ("優しいおばあちゃん", "A gentle elderly female voice, warm and comforting like a grandmother."),
    ("力強いアナウンサー", "A powerful male announcer voice, confident and authoritative."),
    ("かわいい少女", "A cute young girl voice, sweet and innocent with a playful tone."),
]


def generate_audio_with_design(
    text: str,
    voice_description: str,
    language: str,
    speed: float,
) -> tuple[Any, str]:
    """ボイスデザインで音声を生成する。

    Args:
        text: 読み上げテキスト
        voice_description: ボイスの説明
        language: 言語
        speed: 速度

    Returns:
        tuple: (音声ファイルパス, ステータスメッセージ)
    """
    if not text.strip():
        return None, "テキストを入力してください。"

    if not voice_description.strip():
        return None, "ボイスの説明を入力してください。"

    try:
        from mac.engine import DualEngine, TaskType

        logger.info(f"ボイスデザイン生成開始: description={voice_description[:50]}...")
        start_time = time.time()

        # エンジン取得
        engine = DualEngine()

        # 生成
        result = engine.generate(
            text=text,
            task_type=TaskType.VOICE_DESIGN,
            language=language,
            voice_description=voice_description,
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
        logger.error(f"ボイスデザイン生成エラー: {e}")
        return None, f"エラー: {str(e)}"


def create_voice_design_tab() -> None:
    """ボイスデザインタブを作成する。"""
    gr.Markdown(
        """
        ### ボイスデザイン
        
        テキストでボイスの特徴を説明すると、その特徴に合った声で音声を生成します。
        英語で記述するとより正確に特徴が反映されます。
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            # テキスト入力
            text_input = gr.Textbox(
                label="読み上げテキスト",
                placeholder="ここに読み上げたいテキストを入力してください...",
                lines=5,
                max_lines=10,
            )

            # ボイス説明
            voice_description = gr.Textbox(
                label="ボイスの説明",
                placeholder="例: A calm middle-aged male voice with a warm tone.",
                lines=3,
                info="声の特徴を英語で記述してください（日本語でも可）",
            )

            with gr.Row():
                # 言語選択
                language_selector = gr.Dropdown(
                    choices=LANGUAGE_CHOICES,
                    value="Japanese",
                    label="言語",
                    info="出力音声の言語を選択",
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

    # サンプル
    gr.Markdown("### ボイスデザインのサンプル")
    with gr.Row():
        for name_ja, desc_en in VOICE_DESIGN_SAMPLES[:3]:
            with gr.Column():
                gr.Markdown(f"**{name_ja}**")
                sample_btn = gr.Button(f"使用する", size="sm")

                # クロージャで値をキャプチャ
                def make_handler(d: str):
                    return lambda: d
                sample_btn.click(fn=make_handler(desc_en), outputs=[voice_description])

    with gr.Row():
        for name_ja, desc_en in VOICE_DESIGN_SAMPLES[3:]:
            with gr.Column():
                gr.Markdown(f"**{name_ja}**")
                sample_btn = gr.Button(f"使用する", size="sm")

                def make_handler(d: str):
                    return lambda: d
                sample_btn.click(fn=make_handler(desc_en), outputs=[voice_description])

    # サンプルテキスト
    gr.Examples(
        examples=[
            ["こんにちは、今日はいい天気ですね。", "A warm and friendly female voice."],
            ["本日の天気予報をお伝えします。", "A professional male announcer voice."],
            ["むかしむかし、あるところに...", "A gentle storyteller voice, perfect for bedtime stories."],
        ],
        inputs=[text_input, voice_description],
        label="サンプル組み合わせ",
    )

    # イベントハンドラ
    generate_btn.click(
        fn=generate_audio_with_design,
        inputs=[text_input, voice_description, language_selector, speed_slider],
        outputs=[audio_output, status_output],
    )
