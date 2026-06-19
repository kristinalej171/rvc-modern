"""
Pitch extractors с поддержкой GPU-ускорения.
Алгоритмы:
- PM (Parselmouth) — CPU, быстрый
- Harvest — CPU, высокое качество баса
- RMVPE — GPU, лучший по соотношению скорость/качество
- FCPE — GPU, современный, хорошее качество
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from rvc.config import RVCConfig

logger = logging.getLogger(__name__)


class PitchMethod(str, Enum):
    """Доступные методы извлечения F0."""
    PM = "pm"
    HARVEST = "harvest"
    RMVPE = "rmvpe"
    FCPE = "fcpe"


class PitchExtractor(ABC):
    """Абстрактный базовый класс для pitch extractors."""

    def __init__(self, config: RVCConfig) -> None:
        self.config = config
        self.sampling_rate = config.audio.sample_rate
        self.hop_length = config.audio.hop_length
        self.f0_min = int(config.audio.f0_min)
        self.f0_max = int(config.audio.f0_max)

    @abstractmethod
    def compute_f0(
        self,
        wav: np.ndarray,
        p_len: int | None = None,
    ) -> np.ndarray:
        """Вычислить F0 контур."""
        ...

    def compute_f0_uv(
        self,
        wav: np.ndarray,
        p_len: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Вычислить F0 и unvoiced mask."""
        f0 = self.compute_f0(wav, p_len)
        uv = (f0 == 0).astype(np.float32)
        return f0, uv


class PMPitchExtractor(PitchExtractor):
    """Parselmouth-based pitch extractor (CPU, быстрый)."""

    def compute_f0(
        self,
        wav: np.ndarray,
        p_len: int | None = None,
    ) -> np.ndarray:
        import parselmouth

        if p_len is None:
            p_len = wav.shape[0] // self.hop_length

        time_step = self.hop_length / self.sampling_rate
        f0 = (
            parselmouth.Sound(wav, self.sampling_rate)
            .to_pitch_ac(
                time_step=time_step,
                voicing_threshold=0.6,
                pitch_floor=self.f0_min,
                pitch_ceiling=self.f0_max,
            )
            .selected_array["frequency"]
        )

        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.pad(
                f0,
                [[max(0, pad_size), max(0, p_len - len(f0) - pad_size)]],
                mode="constant",
            )

        return f0[:p_len]


class HarvestPitchExtractor(PitchExtractor):
    """Harvest pitch extractor (CPU, высокое качество баса)."""

    def compute_f0(
        self,
        wav: np.ndarray,
        p_len: int | None = None,
    ) -> np.ndarray:
        import pyworld

        if p_len is None:
            p_len = wav.shape[0] // self.hop_length

        f0, t = pyworld.harvest(
            wav.astype(np.float64),
            fs=self.sampling_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        f0 = pyworld.stonemask(
            wav.astype(np.float64), f0, t, self.sampling_rate
        )

        return f0[:p_len]


class RMVPEPitchExtractor(PitchExtractor):
    """
    RMVPE pitch extractor (GPU, лучший по соотношению скорость/качество).
    Использует предобученную модель RMVPE от RVC Project.
    """

    def __init__(
        self,
        config: RVCConfig,
        model_path: str | None = None,
    ) -> None:
        super().__init__(config)
        self.device = torch.device(config.inference.device)
        self.is_half = (
            config.inference.dtype == "float16"
            and self.device.type == "cuda"
        )

        # Путь к модели RMVPE
        if model_path is None:
            model_path = str(config.paths.rmvpe / "rmvpe.pt")

        try:
            # 🔧 BUG FIX: Исправлен неверный путь импорта.
            # Файл rmvpe.py находится в infer/lib/rmvpe.py, а НЕ в rvc/lib/rmvpe.py.
            # Старый импорт `from rvc.lib.rmvpe import RMVPE` приводил к
            # ModuleNotFoundError для всех пользователей, выбравших RMVPE.
            from infer.lib.rmvpe import RMVPE as RMVPEModel

            logger.info("Загрузка RMVPE из %s на устройство %s", model_path, self.device)
            self.model = RMVPEModel(
                model_path,
                is_half=self.is_half,
                device=str(self.device),
            )
        except ImportError as e:
            raise RuntimeError(
                "RMVPE недоступен. Убедитесь, что:\n"
                "  1. Файл infer/lib/rmvpe.py существует\n"
                "  2. Модель assets/rmvpe/rmvpe.pt скачана\n"
                "  3. Зависимости установлены (torch, numpy, scipy)\n"
                f"Исходная ошибка: {e}"
            ) from e
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Файл модели RMVPE не найден: {model_path}\n"
                f"Запустите: python check_and_download_models.py"
            ) from e

    def compute_f0(
        self,
        wav: np.ndarray,
        p_len: int | None = None,
    ) -> np.ndarray:
        """
        Вычислить F0 через RMVPE.
        Args:
            wav: Аудио signal в float32, нормализованный.
            p_len: Целевая длина F0. Если None — вычисляется автоматически.
        Returns:
            F0 массив длины p_len.
        """
        f0 = self.model.infer_from_audio(wav, thred=0.03)

        if p_len is not None and len(f0) != p_len:
            # Интерполяция до нужной длины
            f0 = np.interp(
                np.linspace(0, len(f0) - 1, p_len),
                np.arange(len(f0)),
                f0,
            )

        return f0


class FCPEPitchExtractor(PitchExtractor):
    """
    FCPE pitch extractor (GPU, современный алгоритм).
    Использует библиотеку torchfcpe.
    """

    def __init__(self, config: RVCConfig) -> None:
        super().__init__(config)
        self.device = torch.device(config.inference.device)

        try:
            from torchfcpe import spawn_bundled_infer_model

            logger.info("Загрузка FCPE модели на %s", self.device)
            self.model = spawn_bundled_infer_model(device=self.device)
        except ImportError as e:
            raise RuntimeError(
                "torchfcpe не установлен. Установите: pip install torchfcpe"
            ) from e

    @torch.inference_mode()
    def compute_f0(
        self,
        wav: np.ndarray,
        p_len: int | None = None,
    ) -> np.ndarray:
        """Вычислить F0 через FCPE."""
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)

        f0 = self.model.infer(
            wav_tensor,
            sr=self.sampling_rate,
            decoder_mode="local_argmax",
            threshold=0.006,
            f0_min=self.f0_min,
            f0_max=self.f0_max,
        )

        f0 = f0.squeeze().cpu().numpy()

        if p_len is not None and len(f0) != p_len:
            f0 = np.interp(
                np.linspace(0, len(f0) - 1, p_len),
                np.arange(len(f0)),
                f0,
            )

        return f0


def create_pitch_extractor(
    method: PitchMethod | str,
    config: RVCConfig,
    fallback: bool = True,
) -> PitchExtractor:
    """
    Фабрика pitch extractors с поддержкой fallback.
    Args:
        method: Метод извлечения F0.
        config: RVC конфигурация.
        fallback: Пытаться ли использовать fallback при ошибке.
    Returns:
        Инициализированный pitch extractor.
    """
    if isinstance(method, str):
        method = PitchMethod(method)

    try:
        if method == PitchMethod.PM:
            return PMPitchExtractor(config)
        elif method == PitchMethod.HARVEST:
            return HarvestPitchExtractor(config)
        elif method == PitchMethod.RMVPE:
            return RMVPEPitchExtractor(config)
        elif method == PitchMethod.FCPE:
            return FCPEPitchExtractor(config)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
    except (RuntimeError, ImportError) as e:
        if fallback and method != PitchMethod.PM:
            logger.warning(
                "Не удалось создать %s: %s. Fallback на PM.",
                method.value,
                e,
            )
            return PMPitchExtractor(config)
        raise


def transpose_f0(f0: np.ndarray, key_shift: int) -> np.ndarray:
    """
    Транспонировать F0 на заданное число полутонов.
    Args:
        f0: Массив F0.
        key_shift: Сдвиг в полутонах (+12 = октава вверх).
    Returns:
        Транспонированный F0.
    """
    if key_shift == 0:
        return f0

    # Маска для voiced regions
    voiced = f0 > 0
    result = f0.copy()
    result[voiced] = f0[voiced] * (2 ** (key_shift / 12))

    return result


def f0_to_coarse(
    f0: np.ndarray,
    f0_min: float = 50.0,
    f0_max: float = 1100.0,
) -> np.ndarray:
    """
    Конвертация F0 в coarse (quantized) представление.
    Returns:
        Integer array 1-255 (0 = unvoiced).
    """
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    # Избегаем log(0)
    safe_f0 = np.where(f0 > 0, f0, 1.0)
    f0_mel = 1127 * np.log(1 + safe_f0 / 700)

    f0_mel = np.where(
        f0 > 0,
        (f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1,
        0,
    )

    f0_mel = np.clip(f0_mel, 0, 255)

    return np.rint(f0_mel).astype(np.int32)