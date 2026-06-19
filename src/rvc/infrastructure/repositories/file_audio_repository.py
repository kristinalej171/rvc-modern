"""File-based реализация AudioRepository."""

import logging
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

from rvc.core.domain.entities import AudioFile
from rvc.core.domain.repositories import AudioRepository
from rvc.core.exceptions import AudioProcessingError
from rvc.core.utils import run_ffmpeg

logger = logging.getLogger(__name__)


class FileAudioRepository(AudioRepository):
    """Репозиторий аудио файлов на основе файловой системы."""

    SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

    def load(self, path: Path, target_sr: Optional[int] = None) -> AudioFile:
        """Загрузить аудио файл."""
        if not path.exists():
            raise AudioProcessingError(str(path), "File does not exist")

        try:
            # Загружаем аудио
            audio_data, sr = librosa.load(str(path), sr=target_sr, mono=True)

            # Получаем информацию о файле
            info = sf.info(str(path))

            return AudioFile(
                path=path,
                sample_rate=sr,
                duration=len(audio_data) / sr,
                channels=info.channels,
                format=info.format,
            )
        except Exception as e:
            raise AudioProcessingError(str(path), str(e)) from e

    def save(
        self,
        path: Path,
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> None:
        """Сохранить аудио файл."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            # Нормализация если нужно
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val

            sf.write(str(path), audio_data, sample_rate, subtype="PCM_16")
            logger.info("Saved audio to %s", path)
        except Exception as e:
            raise AudioProcessingError(str(path), str(e)) from e

    def get_files_in_directory(self, directory: Path) -> list[AudioFile]:
        """Получить все аудио файлы в директории."""
        if not directory.exists():
            return []

        audio_files = []
        for file_path in directory.iterdir():
            if file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    audio_file = self.load(file_path)
                    audio_files.append(audio_file)
                except Exception as e:
                    logger.error("Failed to load audio %s: %s", file_path, e)

        return audio_files

    def convert_format(
        self, input_path: Path, output_path: Path, format: str
    ) -> None:
        """Конвертировать формат аудио."""
        extra_args = []
        if format == "mp3":
            extra_args = ["-q:a", "2"]
        elif format == "m4a":
            extra_args = ["-c:a", "aac", "-b:a", "192k"]

        run_ffmpeg(input_path, output_path, extra_args)
        logger.info("Converted %s to %s", input_path, output_path)