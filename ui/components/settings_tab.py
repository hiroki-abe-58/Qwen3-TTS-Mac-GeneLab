# coding=utf-8
# Copyright 2026 Qwen3-TTS-Mac-GeneLab Contributors.
# SPDX-License-Identifier: Apache-2.0
"""
設定タブ

エンジン選択、量子化レベル、メモリモニタなどの設定を管理する。
"""

from __future__ import annotations

import logging
from typing import Any

import gradio as gr

from ui.i18n_utils import t

logger = logging.getLogger(__name__)


def get_system_info() -> str:
    """システム情報を取得する。"""
    try:
        from mac.device_utils import get_mac_info

        info = get_mac_info()
        chip = t("settings.system_info.chip")
        mem = t("settings.system_info.memory")
        mps_yes = t("settings.system_info.mps_available")
        mps_no = t("settings.system_info.mps_unavailable")
        unknown = t("settings.system_info.unknown")

        lines = [
            f"### {t('settings.system_info.title')}",
            "",
            f"- **{chip}**: {info.get('chip', unknown)}",
            f"- **{mem}**: {info.get('total_memory_gb', unknown)} GB",
            f"- **macOS**: {info.get('platform_version', unknown)}",
            f"- **Python**: {info.get('python_version', unknown)}",
            f"- **PyTorch**: {info.get('torch_version', unknown)}",
            f"- **MPS**: {mps_yes if info.get('mps_available') else mps_no}",
        ]
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"システム情報取得エラー: {e}")
        return f"{t('messages.error')}: {e}"


def get_memory_usage() -> dict[str, Any]:
    """メモリ使用量を取得する。"""
    try:
        import psutil

        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024 ** 3),
            "used_gb": mem.used / (1024 ** 3),
            "available_gb": mem.available / (1024 ** 3),
            "percent": mem.percent,
        }
    except Exception as e:
        logger.error(f"メモリ情報取得エラー: {e}")
        return {"total_gb": 0, "used_gb": 0, "available_gb": 0, "percent": 0}


def format_memory_display() -> str:
    """メモリ使用量の表示文字列を生成する。"""
    mem = get_memory_usage()
    return (
        f"{mem['used_gb']:.1f} GB / {mem['total_gb']:.1f} GB "
        f"({mem['percent']:.1f}%) | "
        f"Free: {mem['available_gb']:.1f} GB"
    )


def get_engine_status() -> str:
    """エンジンの状態を取得する。"""
    try:
        from mac.engine import DualEngine

        engine = DualEngine()
        status = engine.get_status()

        lines = [f"### {t('settings.engine.title')}", ""]

        for name, eng_status in status.items():
            loaded = "Loaded" if eng_status.is_loaded else "Not loaded"
            model = eng_status.model_name or "-"
            lines.append(f"**{name.upper()}**")
            lines.append(f"- Status: {loaded}")
            lines.append(f"- Model: {model}")
            lines.append(f"- Device: {eng_status.device}")
            lines.append(f"- dtype: {eng_status.dtype}")
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"エンジン状態取得エラー: {e}")
        return f"{t('messages.error')}: {e}"


def change_engine(engine_type: str) -> str:
    """優先エンジンを変更する。"""
    try:
        from mac.engine import DualEngine

        engine = DualEngine()
        engine.set_preferred_engine(engine_type)
        return f"Engine changed to: {engine_type}"
    except Exception as e:
        logger.error(f"エンジン変更エラー: {e}")
        return f"{t('messages.error')}: {e}"


def unload_models() -> str:
    """全モデルをアンロードする。"""
    try:
        from mac.engine import DualEngine

        engine = DualEngine()
        engine.unload()
        return "All models unloaded."
    except Exception as e:
        logger.error(f"モデルアンロードエラー: {e}")
        return f"{t('messages.error')}: {e}"


def create_settings_tab() -> None:
    """設定タブを作成する。"""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(f"## {t('settings.engine.title')}")

            engine_selector = gr.Dropdown(
                choices=[
                    (t("settings.engine.selector.options.auto"), "auto"),
                    (t("settings.engine.selector.options.mlx"), "mlx"),
                    (t("settings.engine.selector.options.pytorch_mps"), "pytorch_mps"),
                ],
                value="auto",
                label=t("settings.engine.selector.label"),
                info=t("settings.engine.selector.info"),
                interactive=True,
            )

            engine_status_btn = gr.Button(t("settings.engine.status_button"), variant="secondary")
            engine_change_btn = gr.Button(t("settings.engine.change_button"), variant="primary")

            engine_status_display = gr.Markdown(get_engine_status())

            engine_change_result = gr.Textbox(
                label=t("settings.engine.result"),
                interactive=False,
                visible=True,
            )

        with gr.Column(scale=1):
            gr.Markdown(f"## {t('settings.memory.title')}")

            memory_display = gr.Textbox(
                label=t("settings.memory.usage_label"),
                value=format_memory_display(),
                interactive=False,
            )

            memory_refresh_btn = gr.Button(t("settings.memory.refresh_button"), variant="secondary")

            gr.Markdown(f"### {t('settings.memory.model_management')}")
            unload_btn = gr.Button(
                t("settings.memory.unload_button"),
                variant="stop",
            )
            unload_result = gr.Textbox(
                label=t("settings.engine.result"),
                interactive=False,
            )

            gr.Markdown(
                f"""
                #### {t("settings.memory.estimates.title")}

                | Model | dtype | Size |
                |-------|-------|------|
                | 1.7B | bf16 | ~3.4 GB |
                | 1.7B | 8bit | ~1.7 GB |
                | 1.7B | 4bit | ~0.9 GB |
                | 0.6B | bf16 | ~1.2 GB |

                > {t("settings.memory.estimates.note")}
                """
            )

    with gr.Row():
        with gr.Column():
            system_info = gr.Markdown(get_system_info())
            refresh_system_btn = gr.Button(t("settings.system_info.refresh_button"), variant="secondary")

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                f"""
                ## {t("settings.technical_info.title")}

                ### MPS Limitations

                - **Voice Clone**: `float32` required (float16 causes errors)
                - **FlashAttention 2**: Not supported on Mac (uses SDPA)
                - **BFloat16**: May be unstable on M1/M2

                ### Environment Variables

                ```
                PYTORCH_ENABLE_MPS_FALLBACK=1
                PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
                TOKENIZERS_PARALLELISM=false
                ```
                """
            )

    # イベントハンドラ
    engine_status_btn.click(fn=get_engine_status, outputs=[engine_status_display])
    engine_change_btn.click(fn=change_engine, inputs=[engine_selector], outputs=[engine_change_result])
    memory_refresh_btn.click(fn=format_memory_display, outputs=[memory_display])
    unload_btn.click(fn=unload_models, outputs=[unload_result])
    refresh_system_btn.click(fn=get_system_info, outputs=[system_info])
