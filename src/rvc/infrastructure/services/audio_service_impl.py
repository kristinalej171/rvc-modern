"""Реализация AudioService."""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from rvc.core.domain.entities import AudioFile
from rvc.core.domain.repositories import AudioRepository
from rvc.core.domain.services import AudioService

logger = logging.getLogger(__name__)


class AudioServiceImpl(AudioService):
    """Реализация сервиса работы с аудио."""

    def __init__(self, audio_repository: AudioRepository) -> None:
        self._audio_repository = audio_repository

    def load_audio(
        self, path: Path, target_sr: Optional[int] = None
    ) -> AudioFile:
        """Загрузить мета-информацию об аудио."""
        return self._audio_repository.load(path, target_sr)

    def save_audio(
        self, 
        path: Path, 
        audio_data: Union[np.ndarray, torch.Tensor], 
        sample_rate: int
    ) -> None:
        """Сохранить аудио."""
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.detach().cpu().numpy()
            
        self._audio_repository.save(path, audio_data, sample_rate)

    def convert_format(
        self, input_path: Path, output_path: Path, format: str
    ) -> None:
        """Конвертировать формат."""
        self._audio_repository.convert_format(input_path, output_path, format)

    def get_audio_info(self, path: Path) -> AudioFile:
        """Получить информацию об аудио."""
        return self._audio_repository.load(path)