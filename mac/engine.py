# coding=utf-8
# Copyright 2026 Qwen3-TTS-Mac-GeneLab Contributors.
# SPDX-License-Identifier: Apache-2.0
"""
デュアルエンジンマネージャー

- MLX (mlx-audio): デフォルトエンジン。Apple Silicon ネイティブ最適化。
  量子化モデル (4bit/8bit) 対応。メモリ効率が良い。
- PyTorch MPS: フォールバックエンジン。元の qwen_tts パッケージをそのまま使う。
  Voice Clone で float32 が必要な場合や、MLX 非対応機能で使用。

選択ロジック:
  1. ユーザーが明示的に指定 → その通り
  2. Voice Clone → PyTorch MPS (float32) ※MLX の Voice Clone が安定していれば MLX も可
  3. CustomVoice / VoiceDesign → MLX 優先（高速）
  4. MLX ロード失敗 → PyTorch MPS にフォールバック
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np

from .device_utils import (
    TaskType,
    get_attn_implementation,
    get_optimal_device,
    get_optimal_dtype,
    is_mps_available,
)

logger = logging.getLogger(__name__)


class EngineType(str, Enum):
    """推論エンジンの種類"""
    MLX = "mlx"
    PYTORCH_MPS = "pytorch_mps"
    AUTO = "auto"


@dataclass
class GenerationResult:
    """音声生成結果"""
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    engine_used: EngineType
    generation_time_seconds: float
    tokens_per_second: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineStatus:
    """エンジンの状態"""
    engine_type: EngineType
    is_loaded: bool
    model_name: str | None
    device: str
    dtype: str
    memory_used_mb: float | None = None


class MLXEngine:
    """MLX ベースの TTS エンジン（mlx-audio 使用）

    新旧両方の mlx-audio API に自動対応:
    - Modern API (>=0.3.x): load_model() → nn.Module, model.generate() ジェネレータ
    - Legacy API (<0.3.x): load_model() → (model, processor), 外部 generate() 関数
    """

    _LANG_MAP: dict[str, str] = {
        "japanese": "japanese",
        "english": "english",
        "chinese": "chinese",
        "korean": "korean",
        "french": "french",
        "german": "german",
        "spanish": "spanish",
        "italian": "italian",
        "portuguese": "portuguese",
        "russian": "russian",
    }

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._model_name: str | None = None
        self._mlx_available: bool = False
        self._use_legacy_api: bool = False
        self._generate_fn: Any = None
        self._supported_speakers: list[str] = []

        try:
            import mlx.core as mx  # noqa: F401
            self._mlx_available = True
            logger.info("MLX エンジンが利用可能です。")
        except ImportError:
            logger.warning("MLX がインストールされていません。")

    @property
    def is_available(self) -> bool:
        return self._mlx_available

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def supported_speakers(self) -> list[str]:
        return self._supported_speakers

    def _to_lang_code(self, language: str) -> str:
        return self._LANG_MAP.get(language.lower(), "auto")

    def _resolve_speaker(self, speaker: str) -> str:
        """UI のスピーカー名をモデルが受け付ける名前に解決する。"""
        if not self._supported_speakers:
            return speaker.lower()

        lower = speaker.lower()
        if lower in self._supported_speakers:
            return lower

        for s in self._supported_speakers:
            if s == speaker:
                return s

        logger.warning(
            f"スピーカー '{speaker}' が見つかりません。"
            f"利用可能: {self._supported_speakers} → 先頭を使用"
        )
        return self._supported_speakers[0]

    def load_model(
        self,
        model_name: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    ) -> None:
        """MLX モデルをロードする。新旧両方の mlx-audio API に自動対応。"""
        if not self._mlx_available:
            raise RuntimeError("MLX が利用できません。")

        if self._model is not None and self._model_name == model_name:
            logger.info(f"モデル '{model_name}' は既にロードされています。")
            return

        if self._model is not None:
            self.unload()

        logger.info(f"MLX モデルをロード中: {model_name}")
        start_time = time.time()

        try:
            from mlx_audio.tts.utils import load_model as mlx_load_model

            loaded = mlx_load_model(model_name)

            if isinstance(loaded, tuple):
                # Legacy API: (model, processor) を返す旧バージョン
                self._model, self._processor = loaded
                self._use_legacy_api = True
                try:
                    from mlx_audio.tts import generate as gen_module
                    self._generate_fn = gen_module
                except ImportError:
                    self._generate_fn = None
                logger.info("MLX Legacy API を使用します。")
            else:
                # Modern API: 単一の nn.Module を返す
                self._model = loaded
                self._processor = None
                self._use_legacy_api = False
                self._generate_fn = None

            # サポートスピーカーを取得
            if hasattr(self._model, "get_supported_speakers"):
                self._supported_speakers = self._model.get_supported_speakers()
                logger.info(f"MLX サポートスピーカー: {self._supported_speakers}")

            self._model_name = model_name
            elapsed = time.time() - start_time
            logger.info(f"MLX モデルのロード完了: {elapsed:.2f}秒")

        except Exception as e:
            logger.error(f"MLX モデルのロードに失敗: {e}")
            raise

    def generate(
        self,
        text: str,
        language: str = "Japanese",
        speaker: str = "Chelsie",
        emotion: str | None = None,
        voice_description: str | None = None,
        speed: float = 1.0,
        progress_callback: Callable[[float], None] | None = None,
    ) -> tuple[np.ndarray, int]:
        """音声を生成する。新旧両API に対応。"""
        if not self.is_loaded:
            raise RuntimeError("モデルがロードされていません。先に load_model() を呼んでください。")

        lang_code = self._to_lang_code(language)
        resolved_speaker = self._resolve_speaker(speaker)
        logger.info(
            f"MLX で音声生成中: '{text[:50]}...' "
            f"(lang_code: {lang_code}, voice: {resolved_speaker})"
        )

        try:
            if self._use_legacy_api:
                return self._generate_legacy(
                    text, language, resolved_speaker, emotion,
                    voice_description, speed,
                )
            return self._generate_modern(
                text, lang_code, resolved_speaker, emotion,
                voice_description, speed,
            )
        except Exception as e:
            logger.error(f"MLX 音声生成エラー: {e}")
            raise

    def _generate_legacy(
        self,
        text: str,
        language: str,
        speaker: str,
        emotion: str | None,
        voice_description: str | None,
        speed: float,
    ) -> tuple[np.ndarray, int]:
        """Legacy API (mlx-audio <0.3.x) で音声を生成する。"""
        if self._generate_fn is None:
            raise RuntimeError("Legacy generate 関数が利用できません。")

        gen_fn = getattr(self._generate_fn, "generate", self._generate_fn)
        if callable(gen_fn):
            audio = gen_fn(
                self._model,
                self._processor,
                text,
                language=language,
                speaker=speaker,
                verbose=False,
            )
        else:
            raise RuntimeError("mlx-audio の generate 関数が見つかりません。")

        sample_rate = 24000
        if hasattr(audio, "tolist"):
            audio = np.array(audio.tolist(), dtype=np.float32)
        elif not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)

        return audio, sample_rate

    def _generate_modern(
        self,
        text: str,
        lang_code: str,
        speaker: str,
        emotion: str | None,
        voice_description: str | None,
        speed: float,
    ) -> tuple[np.ndarray, int]:
        """Modern API (mlx-audio >=0.3.x) で音声を生成する。"""
        instruct = emotion or voice_description or None

        gen_kwargs: dict[str, Any] = {
            "text": text,
            "voice": speaker,
            "lang_code": lang_code,
            "speed": speed,
            "verbose": False,
        }
        if instruct:
            gen_kwargs["instruct"] = instruct

        audio_chunks: list[np.ndarray] = []
        sample_rate = getattr(self._model, "sample_rate", 24000)

        for result in self._model.generate(**gen_kwargs):
            chunk = result.audio
            if hasattr(chunk, "tolist"):
                chunk = np.array(chunk.tolist(), dtype=np.float32)
            elif not isinstance(chunk, np.ndarray):
                chunk = np.array(chunk, dtype=np.float32)
            audio_chunks.append(chunk)
            if hasattr(result, "sample_rate") and result.sample_rate:
                sample_rate = result.sample_rate

        if not audio_chunks:
            raise RuntimeError("音声が生成されませんでした。")

        audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
        return audio, sample_rate

    def unload(self) -> None:
        """モデルをアンロードしてメモリを解放する。"""
        if self._model is not None:
            logger.info("MLX モデルをアンロード中...")
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._model_name = None
            self._use_legacy_api = False
            self._generate_fn = None
            self._supported_speakers = []

            try:
                import mlx.core as mx
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
            except Exception:
                pass

            gc.collect()
            logger.info("MLX モデルのアンロード完了")

    def get_status(self) -> EngineStatus:
        return EngineStatus(
            engine_type=EngineType.MLX,
            is_loaded=self.is_loaded,
            model_name=self._model_name,
            device="metal",
            dtype="auto (量子化)",
        )


class PyTorchMPSEngine:
    """PyTorch MPS ベースの TTS エンジン（qwen_tts 使用）"""

    def __init__(self) -> None:
        self._model: Any = None
        self._model_name: str | None = None
        self._device: str = get_optimal_device()
        self._task_type: TaskType | None = None
        self._dtype: Any = None

    @property
    def is_available(self) -> bool:
        """PyTorch MPS が利用可能かどうか"""
        return is_mps_available() or self._device in ("cuda", "cpu")

    @property
    def is_loaded(self) -> bool:
        """モデルがロードされているかどうか"""
        return self._model is not None

    def load_model(
        self,
        model_name: str | None = None,
        task_type: TaskType = TaskType.CUSTOM_VOICE,
    ) -> None:
        """PyTorch モデルをロードする。

        Args:
            model_name: HuggingFace モデル名（None の場合はタスクに応じて自動選択）
            task_type: タスクの種類（dtype 選択に使用）
        """
        # モデル名の自動選択
        if model_name is None:
            if task_type == TaskType.CUSTOM_VOICE:
                model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            elif task_type == TaskType.VOICE_DESIGN:
                model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
            elif task_type == TaskType.VOICE_CLONE:
                model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            else:
                model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

        if self._model is not None and self._model_name == model_name:
            logger.info(f"モデル '{model_name}' は既にロードされています。")
            return

        # 既存モデルをアンロード
        if self._model is not None:
            self.unload()

        logger.info(f"PyTorch MPS モデルをロード中: {model_name}")
        start_time = time.time()

        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            dtype = get_optimal_dtype(task_type, self._device)
            attn_impl = get_attn_implementation()

            logger.info(f"  モデル: {model_name}")
            logger.info(f"  dtype: {dtype}")
            logger.info(f"  device: {self._device}")
            logger.info(f"  attention: {attn_impl}")

            # Voice Clone は CPU で実行（MPS の Placeholder storage 問題回避）
            # それ以外のタスクは MPS に移動を試みる
            use_cpu = (task_type == TaskType.VOICE_CLONE)

            self._model = Qwen3TTSModel.from_pretrained(
                model_name,
                dtype=dtype,
                attn_implementation=attn_impl,
            )

            if not use_cpu and self._device == "mps":
                try:
                    self._model.model = self._model.model.to(self._device)
                except RuntimeError as move_err:
                    logger.warning(f"MPS への移動に失敗、CPU で実行します: {move_err}")
                    use_cpu = True

            if use_cpu:
                logger.info("Voice Clone: CPU で実行します。")
                self._model.model = self._model.model.to("cpu")
                # デバイス参照を更新
                self._model.device = torch.device("cpu")

            self._model_name = model_name
            self._dtype = dtype
            self._task_type = task_type

            elapsed = time.time() - start_time
            logger.info(f"PyTorch MPS モデルのロード完了: {elapsed:.2f}秒")

        except ImportError as e:
            error_msg = (
                f"必要なパッケージがインストールされていません: {e}\n"
                "解決策: pip install -e . を実行してください。"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except OSError as e:
            if "No space left" in str(e) or "disk" in str(e).lower():
                error_msg = (
                    f"ディスク容量が不足しています: {e}\n"
                    f"モデル '{model_name}' のダウンロードには約4GB必要です。\n"
                    "解決策: 不要なファイルを削除してディスク容量を確保してください。"
                )
            else:
                error_msg = (
                    f"モデルファイルの読み込みに失敗しました: {e}\n"
                    "解決策: ネットワーク接続を確認し、再試行してください。"
                )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except Exception as e:
            error_str = str(e)
            # よくあるエラーパターンを検出
            if "CUDA" in error_str or "cuda" in error_str:
                hint = "ヒント: このプロジェクトは Apple Silicon Mac 向けです。CUDA は使用できません。"
            elif "out of memory" in error_str.lower() or "oom" in error_str.lower():
                hint = (
                    "ヒント: メモリ不足です。他のアプリを閉じるか、"
                    "量子化モデル (8bit) を使用してください。"
                )
            elif "connection" in error_str.lower() or "network" in error_str.lower():
                hint = "ヒント: ネットワーク接続を確認してください。"
            else:
                hint = f"モデル: {model_name}, タスク: {task_type.value}"

            error_msg = f"PyTorch MPS モデルのロードに失敗しました: {e}\n{hint}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def generate(
        self,
        text: str,
        language: str = "Japanese",
        speaker: str = "Chelsie",
        task_type: TaskType = TaskType.CUSTOM_VOICE,
        emotion: str | None = None,
        voice_description: str | None = None,
        reference_audio: np.ndarray | None = None,
        reference_text: str | None = None,
        reference_sr: int = 24000,
        speed: float = 1.0,
        progress_callback: Callable[[float], None] | None = None,
    ) -> tuple[np.ndarray, int]:
        """音声を生成する。

        Args:
            text: 読み上げるテキスト
            language: 言語
            speaker: スピーカー名
            task_type: タスクの種類
            emotion: 感情指示
            voice_description: ボイスデザイン記述
            reference_audio: 参照音声（Voice Clone 用）
            reference_text: 参照音声のテキスト（Voice Clone 用）
            reference_sr: 参照音声のサンプルレート
            speed: 速度
            progress_callback: 進捗コールバック

        Returns:
            tuple[np.ndarray, int]: (音声データ, サンプルレート)
        """
        if not self.is_loaded:
            raise RuntimeError("モデルがロードされていません。先に load_model() を呼んでください。")

        # タスクタイプが異なる場合は自動的にリロード
        if hasattr(self, "_task_type") and self._task_type != task_type:
            logger.info(f"タスクタイプが変更されました ({self._task_type} -> {task_type})。モデルを再ロードします。")
            self.load_model(model_name=None, task_type=task_type)

        logger.info(f"PyTorch MPS で音声生成中: '{text[:50]}...'")

        try:
            # CustomVoice の場合
            if task_type == TaskType.CUSTOM_VOICE:
                # 感情指示を組み込み
                instruct = emotion if emotion else None

                wavs, sr = self._model.generate_custom_voice(
                    text=text,
                    speaker=speaker,
                    language=language,
                    instruct=instruct,
                )
                audio = wavs[0] if isinstance(wavs, list) else wavs

            # VoiceDesign の場合
            elif task_type == TaskType.VOICE_DESIGN:
                if not voice_description:
                    raise ValueError(
                        "VoiceDesign には voice_description (声の説明) が必要です。\n"
                        "例: 'A calm middle-aged male voice with a warm tone.'"
                    )

                wavs, sr = self._model.generate_voice_design(
                    text=text,
                    instruct=voice_description,
                    language=language,
                )
                audio = wavs[0] if isinstance(wavs, list) else wavs

            # Voice Clone の場合
            elif task_type == TaskType.VOICE_CLONE:
                if reference_audio is None:
                    raise ValueError(
                        "Voice Clone には参照音声 (reference_audio) が必要です。\n"
                        "3秒以上のクリアな音声ファイルをアップロードしてください。"
                    )
                if reference_text is None:
                    raise ValueError(
                        "Voice Clone には参照音声のテキスト (reference_text) が必要です。\n"
                        "Whisper で自動書き起こしするか、手動で入力してください。"
                    )

                # 参照音声の長さを確認
                ref_duration = len(reference_audio) / reference_sr
                if ref_duration < 1.0:
                    logger.warning(f"参照音声が短すぎます ({ref_duration:.1f}秒)。3秒以上を推奨します。")
                elif ref_duration > 30.0:
                    logger.warning(f"参照音声が長すぎます ({ref_duration:.1f}秒)。10秒程度を推奨します。")

                # 参照音声を (audio, sr) のタプル形式で渡す
                ref_audio_tuple = (reference_audio, reference_sr)

                wavs, sr = self._model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=ref_audio_tuple,
                    ref_text=reference_text,
                    x_vector_only_mode=False,  # ICL モードを使用
                )
                audio = wavs[0] if isinstance(wavs, list) else wavs

            else:
                raise ValueError(f"不明なタスクタイプ: {task_type}")

            # numpy 配列に変換
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)

            logger.info(f"音声生成完了: {len(audio)/sr:.2f}秒")
            return audio, sr

        except ValueError:
            # ValueError はそのまま再送出（ユーザー入力エラー）
            raise

        except RuntimeError as e:
            error_str = str(e)
            if "probability tensor contains" in error_str:
                error_msg = (
                    "Voice Clone で数値エラーが発生しました。\n"
                    "原因: float16/bf16 では Voice Clone が不安定です。\n"
                    "解決策: 自動的に float32 が選択されるはずですが、"
                    "問題が続く場合はモデルを再ロードしてください。"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            elif "out of memory" in error_str.lower():
                error_msg = (
                    "メモリ不足エラーが発生しました。\n"
                    "解決策:\n"
                    "  1. 他のアプリケーションを閉じる\n"
                    "  2. 短いテキストで試す\n"
                    "  3. 設定でモデルをアンロードして再試行"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            else:
                logger.error(f"音声生成中にランタイムエラー: {e}")
                raise

        except Exception as e:
            error_msg = f"音声生成中に予期しないエラーが発生しました: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def unload(self) -> None:
        """モデルをアンロードしてメモリを解放する。"""
        if self._model is not None:
            logger.info("PyTorch MPS モデルをアンロード中...")
            del self._model
            self._model = None
            self._model_name = None

            # MPS キャッシュクリア
            try:
                import torch
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            except Exception:
                pass

            gc.collect()
            logger.info("PyTorch MPS モデルのアンロード完了")

    def get_status(self) -> EngineStatus:
        """エンジンの状態を取得する。"""
        dtype_str = str(getattr(self, "_dtype", "unknown"))
        return EngineStatus(
            engine_type=EngineType.PYTORCH_MPS,
            is_loaded=self.is_loaded,
            model_name=self._model_name,
            device=self._device,
            dtype=dtype_str,
        )


# タスクタイプ → MLX モデルの対応表
_MLX_MODEL_MAP: dict[TaskType, str] = {
    TaskType.CUSTOM_VOICE: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    TaskType.VOICE_DESIGN: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
    TaskType.VOICE_CLONE: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
}


class DualEngine:
    """デュアルエンジンマネージャー（シングルトン）

    MLX と PyTorch MPS を自動的に切り替えて使用する。
    """

    _instance: DualEngine | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> DualEngine:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        engine_type: EngineType | str = EngineType.AUTO,
    ) -> None:
        if self._initialized:
            return

        if isinstance(engine_type, str):
            engine_type = EngineType(engine_type)

        self._preferred_engine = engine_type
        self._mlx_engine = MLXEngine()
        self._pytorch_engine = PyTorchMPSEngine()
        self._current_engine: MLXEngine | PyTorchMPSEngine | None = None
        self._initialized = True

        logger.info(f"DualEngine 初期化: 優先エンジン = {engine_type.value}")

    def _select_engine(self, task_type: TaskType) -> MLXEngine | PyTorchMPSEngine:
        """タスクに応じてエンジンを選択する。

        Args:
            task_type: タスクの種類

        Returns:
            選択されたエンジン
        """
        # 明示的なエンジン指定
        if self._preferred_engine == EngineType.MLX:
            if self._mlx_engine.is_available:
                return self._mlx_engine
            logger.warning("MLX が利用できません。PyTorch MPS にフォールバックします。")
            return self._pytorch_engine

        if self._preferred_engine == EngineType.PYTORCH_MPS:
            return self._pytorch_engine

        # AUTO モード
        # Voice Clone は PyTorch MPS（float32 必須）
        if task_type == TaskType.VOICE_CLONE:
            logger.info("Voice Clone: PyTorch MPS を使用します（float32 必須）。")
            return self._pytorch_engine

        # CustomVoice / VoiceDesign は MLX 優先（高速）
        if self._mlx_engine.is_available:
            logger.info(f"{task_type.value}: MLX を使用します（高速）。")
            return self._mlx_engine

        logger.info(f"{task_type.value}: PyTorch MPS を使用します。")
        return self._pytorch_engine

    def load_model(
        self,
        model_name: str | None = None,
        task_type: TaskType = TaskType.CUSTOM_VOICE,
    ) -> None:
        """モデルをロードする。

        Args:
            model_name: モデル名（None の場合はデフォルト）
            task_type: タスクの種類
        """
        engine = self._select_engine(task_type)

        # 別のエンジンがロード済みの場合はアンロード
        if self._current_engine is not None and self._current_engine is not engine:
            self._current_engine.unload()

        # モデルをロード (model_name が None の場合、タスクに応じて自動選択)
        if isinstance(engine, MLXEngine):
            if model_name is None:
                model_name = _MLX_MODEL_MAP.get(
                    task_type,
                    "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
                )
            engine.load_model(model_name)
        else:
            # PyTorch エンジンは model_name=None の場合、タスクタイプに応じて自動選択
            engine.load_model(model_name, task_type)

        self._current_engine = engine

    def generate(
        self,
        text: str,
        task_type: TaskType = TaskType.CUSTOM_VOICE,
        language: str = "Japanese",
        speaker: str = "Chelsie",
        emotion: str | None = None,
        voice_description: str | None = None,
        reference_audio: np.ndarray | None = None,
        reference_text: str | None = None,
        reference_sr: int = 24000,
        speed: float = 1.0,
        progress_callback: Callable[[float], None] | None = None,
    ) -> GenerationResult:
        """音声を生成する。

        Args:
            text: 読み上げるテキスト
            task_type: タスクの種類
            language: 言語
            speaker: スピーカー名
            emotion: 感情指示
            voice_description: ボイスデザイン記述
            reference_audio: 参照音声
            reference_text: 参照音声のテキスト
            speed: 速度
            progress_callback: 進捗コールバック

        Returns:
            GenerationResult: 生成結果
        """
        # エンジン選択とモデルロード
        engine = self._select_engine(task_type)

        # MLX の場合、タスクタイプごとに異なるモデルが必要
        required_model = _MLX_MODEL_MAP.get(task_type)
        needs_reload = (
            not engine.is_loaded
            or (
                isinstance(engine, MLXEngine)
                and required_model
                and engine._model_name != required_model
            )
        )
        if needs_reload:
            self.load_model(task_type=task_type)
            engine = self._current_engine

        start_time = time.time()

        # 生成
        if isinstance(engine, MLXEngine):
            audio, sample_rate = engine.generate(
                text=text,
                language=language,
                speaker=speaker,
                emotion=emotion,
                voice_description=voice_description,
                speed=speed,
                progress_callback=progress_callback,
            )
            engine_used = EngineType.MLX
        else:
            audio, sample_rate = engine.generate(
                text=text,
                language=language,
                speaker=speaker,
                task_type=task_type,
                emotion=emotion,
                voice_description=voice_description,
                reference_audio=reference_audio,
                reference_text=reference_text,
                reference_sr=reference_sr,
                speed=speed,
                progress_callback=progress_callback,
            )
            engine_used = EngineType.PYTORCH_MPS

        generation_time = time.time() - start_time
        duration = len(audio) / sample_rate

        return GenerationResult(
            audio=audio,
            sample_rate=sample_rate,
            duration_seconds=duration,
            engine_used=engine_used,
            generation_time_seconds=generation_time,
            metadata={
                "text": text,
                "language": language,
                "speaker": speaker,
                "task_type": task_type.value,
            },
        )

    def unload(self) -> None:
        """すべてのモデルをアンロードする。"""
        self._mlx_engine.unload()
        self._pytorch_engine.unload()
        self._current_engine = None

    def get_status(self) -> dict[str, EngineStatus]:
        """エンジンの状態を取得する。"""
        return {
            "mlx": self._mlx_engine.get_status(),
            "pytorch_mps": self._pytorch_engine.get_status(),
        }

    def set_preferred_engine(self, engine_type: EngineType | str) -> None:
        """優先エンジンを設定する。

        Args:
            engine_type: エンジンタイプ
        """
        if isinstance(engine_type, str):
            engine_type = EngineType(engine_type)
        self._preferred_engine = engine_type
        logger.info(f"優先エンジンを変更: {engine_type.value}")


# 利便性のためのエクスポート
__all__ = [
    "EngineType",
    "TaskType",
    "GenerationResult",
    "EngineStatus",
    "MLXEngine",
    "PyTorchMPSEngine",
    "DualEngine",
]
