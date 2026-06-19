"""Обработка аудио: загрузка, сохранение, нормализация."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import librosa
import numpy as np
import soundfile as sf
import torch

from rvc.core.utils import run_ffmpeg, sanitize_path

if TYPE_CHECKING:
    from rvc.config import RVCConfig

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Класс для загрузки и обработки аудио."""

    def __init__(self, config: RVCConfig) -> None:
        self.config = config
        self.sample_rate = config.audio.sample_rate

    def load_audio(self, path: str | Path, target_sr: int | None = None) -> np.ndarray:
        """
        Загрузка аудиофайла в mono float32 numpy array.

        Args:
            path: Путь к аудиофайлу.
            target_sr: Целевая частота дискретизации.
                       Если None, используется self.sample_rate.

        Returns:
            numpy array формы (samples,) с dtype float32.

        Raises:
            FileNotFoundError: Если файл не существует.
            RuntimeError: Если не удалось загрузить аудио.
        """
        path = sanitize_path(path)
        sr = target_sr or self.sample_rate

        if not path.exists():
            raise FileNotFoundError(f"Аудиофайл не найден: {path}")

        try:
            # librosa использует ffmpeg/soundfile под капотом
            audio, _ = librosa.load(str(path), sr=sr, mono=True, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить аудио {path}: {e}") from e

        return audio

    def save_audio(
        self,
        path: str | Path,
        audio: np.ndarray | torch.Tensor,
        sample_rate: int | None = None,
    ) -> None:
        """
        Сохранение аудио в файл.

        Args:
            path: Путь для сохранения.
            audio: Аудио данные (numpy или torch tensor).
            sample_rate: Частота дискретизации.
        """
        path = sanitize_path(path)
        sr = sample_rate or self.sample_rate

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Нормализация, если нужно
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, sr, subtype="PCM_16")
        logger.info("Аудио сохранено: %s", path)

    def convert_format(
        self,
        input_path: Path,
        output_path: Path,
        format: str = "wav",
    ) -> Path:
        """
        Конвертация аудио в другой формат через ffmpeg.

        Args:
            input_path: Входной файл.
            output_path: Выходной файл.
            format: Целевой формат (wav, flac, mp3, m4a).

        Returns:
            Путь к выходному файлу.
        """
        extra_args: list[str] = []
        if format == "mp3":
            extra_args = ["-q:a", "2"]
        elif format == "m4a":
            extra_args = ["-c:a", "aac", "-b:a", "192k"]

        run_ffmpeg(input_path, output_path, extra_args)
        return output_path

    @staticmethod
    def normalize(audio: np.ndarray, max_value: float = 0.95) -> np.ndarray:
        """Нормализация громкости аудио."""
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio * (max_value / peak)
        return audio

    @staticmethod
    def to_int16(audio: np.ndarray) -> np.ndarray:
        """Конвертация float32 [-1, 1] в int16."""
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767).astype(np.int16)

    def resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Ресэмплинг аудио."""
        if orig_sr == target_sr:
            return audio
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)