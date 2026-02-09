# coding=utf-8
# Copyright 2026 Qwen3-TTS-Mac-GeneLab Contributors.
# SPDX-License-Identifier: Apache-2.0
"""
カスタムボイスタブ

9種のプリセットスピーカーを選択し、感情指示と言語を指定して音声を生成する。
"""

from __future__ import annotations

import logging
import tempfile
import time
from typing import Any

import gradio as gr
import numpy as np

logger = logging.getLogger(__name__)

# スピーカー定義（日本語説明付き）
SPEAKERS = {
    "Chelsie": {
        "name_ja": "チェルシー",
        "description": "明るい若い女性の声。エネルギッシュで親しみやすい。",
        "gender": "female",
        "age": "young",
    },
    "Ethan": {
        "name_ja": "イーサン",
        "description": "穏やかな若い男性の声。落ち着いた雰囲気。",
        "gender": "male",
        "age": "young",
    },
    "Aiden": {
        "name_ja": "エイデン",
        "description": "渋みのある男性の声。成熟した安定感。",
        "gender": "male",
        "age": "middle",
    },
    "Bella": {
        "name_ja": "ベラ",
        "description": "温かみのある女性の声。優しく包み込む雰囲気。",
        "gender": "female",
        "age": "middle",
    },
    "Vivian": {
        "name_ja": "ヴィヴィアン",
        "description": "エネルギッシュな女性の声。活発で明るい。",
        "gender": "female",
        "age": "young",
    },
    "Lucas": {
        "name_ja": "ルーカス",
        "description": "明るい若い男性の声。爽やかで親しみやすい。",
        "gender": "male",
        "age": "young",
    },
    "Eleanor": {
        "name_ja": "エレノア",
        "description": "上品な女性の声。落ち着いた知的な雰囲気。",
        "gender": "female",
        "age": "middle",
    },
    "Alexander": {
        "name_ja": "アレクサンダー",
        "description": "力強い男性の声。威厳と安定感。",
        "gender": "male",
        "age": "middle",
    },
    "Serena": {
        "name_ja": "セレナ",
        "description": "癒やしの女性の声。穏やかでリラックス。",
        "gender": "female",
        "age": "young",
    },
}

# 言語選択 (表示ラベル -> エンジン用キー)
LANGUAGE_CHOICES = [
    "Japanese", "English", "Chinese", "Korean",
    "French", "German", "Spanish", "Italian",
    "Portuguese", "Russian",
]

# 感情プリセット (表示ラベル -> エンジン用キー)
EMOTION_CHOICES = [
    "neutral", "happy", "sad", "angry",
    "surprised", "fearful", "disgusted",
    "calm", "excited", "tender",
]


def generate_audio(
    text: str,
    speaker: str,
    language: str,
    emotion: str,
    speed: float,
) -> tuple[Any, str]:
    """音声を生成する。

    Args:
        text: 読み上げテキスト
        speaker: スピーカー名
        language: 言語
        emotion: 感情
        speed: 速度

    Returns:
        tuple: (音声ファイルパス, ステータスメッセージ)
    """
    if not text.strip():
        return None, "テキストを入力してください。"

    try:
        from mac.engine import DualEngine, TaskType

        logger.info(f"音声生成開始: speaker={speaker}, language={language}, emotion={emotion}")
        start_time = time.time()

        # エンジン取得
        engine = DualEngine()

        # 生成
        result = engine.generate(
            text=text,
            task_type=TaskType.CUSTOM_VOICE,
            language=language,
            speaker=speaker,
            emotion=emotion if emotion != "neutral" else None,
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
        logger.error(f"音声生成エラー: {e}")
        return None, f"エラー: {str(e)}"


def create_speaker_selector() -> gr.Radio:
    """スピーカー選択コンポーネントを作成する。"""
    choices = [
        f"{info['name_ja']}（{name}）- {info['description']}"
        for name, info in SPEAKERS.items()
    ]
    return gr.Radio(
        choices=choices,
        value=choices[0],
        label="スピーカー選択",
        info="9種類のプリセットボイスから選択してください",
    )


def extract_speaker_name(selection: str) -> str:
    """選択文字列からスピーカー名を抽出する。"""
    # "チェルシー（Chelsie）- ..." から "Chelsie" を抽出
    import re
    match = re.search(r"（(\w+)）", selection)
    if match:
        return match.group(1)
    return "Chelsie"


def create_custom_voice_tab() -> None:
    """カスタムボイスタブを作成する。"""
    with gr.Row():
        with gr.Column(scale=2):
            # テキスト入力
            text_input = gr.Textbox(
                label="読み上げテキスト",
                placeholder="ここに読み上げたいテキストを入力してください...",
                lines=5,
                max_lines=10,
            )

            # スピーカー選択
            speaker_selector = create_speaker_selector()

            with gr.Row():
                # 言語選択
                language_selector = gr.Dropdown(
                    choices=LANGUAGE_CHOICES,
                    value="Japanese",
                    label="言語",
                    info="出力音声の言語を選択",
                )

                # 感情選択
                emotion_selector = gr.Dropdown(
                    choices=EMOTION_CHOICES,
                    value="neutral",
                    label="感情",
                    info="感情を指定（オプション）",
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

            # サンプルテキスト
            gr.Examples(
                examples=[
                    ["こんにちは、今日はいい天気ですね。散歩に出かけましょう。"],
                    ["本日は、弊社の新製品発表会にお越しいただき、誠にありがとうございます。"],
                    ["むかしむかし、あるところに、おじいさんとおばあさんが住んでいました。"],
                    ["Hello, welcome to Qwen3-TTS. This is a demonstration of voice synthesis."],
                    ["人工知能による音声合成技術は、日々進化を続けています。"],
                ],
                inputs=[text_input],
                label="サンプルテキスト",
            )

    # イベントハンドラ
    def on_generate(text: str, speaker_sel: str, language: str, emotion: str, speed: float) -> tuple:
        speaker = extract_speaker_name(speaker_sel)
        return generate_audio(text, speaker, language, emotion, speed)

    generate_btn.click(
        fn=on_generate,
        inputs=[text_input, speaker_selector, language_selector, emotion_selector, speed_slider],
        outputs=[audio_output, status_output],
    )
