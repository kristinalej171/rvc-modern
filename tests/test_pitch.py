"""
Тесты модуля rvc.core.pitch.
Покрывают:
- transpose_f0
- f0_to_coarse
- PitchMethod enum
- PMPitchExtractor (работает на CPU)
"""
from __future__ import annotations

import numpy as np
import pytest

from rvc.core.pitch import (
    PitchMethod,
    PMPitchExtractor,
    f0_to_coarse,
    transpose_f0,
)


class TestTransposeF0:
    """Тесты transpose_f0."""

    def test_zero_shift(self) -> None:
        """Сдвиг 0 не меняет F0."""
        f0 = np.array([100.0, 200.0, 300.0, 0.0])
        result = transpose_f0(f0, key_shift=0)
        np.testing.assert_array_equal(result, f0)

    def test_octave_up(self) -> None:
        """Сдвиг +12 полутонов = октава вверх (×2)."""
        f0 = np.array([100.0, 200.0, 0.0])
        result = transpose_f0(f0, key_shift=12)
        np.testing.assert_allclose(result[0], 200.0, rtol=1e-6)
        np.testing.assert_allclose(result[1], 400.0, rtol=1e-6)
        assert result[2] == 0.0  # Unvoiced не меняется

    def test_octave_down(self) -> None:
        """Сдвиг -12 полутонов = октава вниз (÷2)."""
        f0 = np.array([200.0, 400.0, 0.0])
        result = transpose_f0(f0, key_shift=-12)
        np.testing.assert_allclose(result[0], 100.0, rtol=1e-6)
        np.testing.assert_allclose(result[1], 200.0, rtol=1e-6)

    def test_semitone_shift(self) -> None:
        """Сдвиг на 1 полутон (×2^(1/12))."""
        f0 = np.array([440.0])  # Нота A4
        result = transpose_f0(f0, key_shift=1)  # A#4
        expected = 440.0 * (2 ** (1 / 12))
        np.testing.assert_allclose(result[0], expected, rtol=1e-6)

    def test_unvoiced_preserved(self) -> None:
        """Unvoiced regions (f0=0) не транспонируются."""
        f0 = np.array([0.0, 0.0, 100.0, 0.0])
        result = transpose_f0(f0, key_shift=12)
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[3] == 0.0
        assert result[2] > 100.0  # Voiced транспонирован

    def test_empty_array(self) -> None:
        """Пустой массив не падает."""
        f0 = np.array([], dtype=np.float64)
        result = transpose_f0(f0, key_shift=12)
        assert len(result) == 0

    def test_does_not_modify_input(self) -> None:
        """Исходный массив не модифицируется."""
        f0 = np.array([100.0, 200.0])
        original = f0.copy()
        transpose_f0(f0, key_shift=12)
        np.testing.assert_array_equal(f0, original)


class TestF0ToCoarse:
    """Тесты f0_to_coarse."""

    def test_basic_conversion(self) -> None:
        """Базовая конвертация."""
        f0 = np.array([100.0, 200.0, 400.0])
        coarse = f0_to_coarse(f0)
        assert coarse.dtype == np.int32
        assert len(coarse) == 3
        assert all(1 <= c <= 255 for c in coarse)

    def test_unvoiced_becomes_zero(self) -> None:
        """Unvoiced (f0=0) превращается в 0."""
        f0 = np.array([100.0, 0.0, 400.0, 0.0])
        coarse = f0_to_coarse(f0)
        assert coarse[1] == 0
        assert coarse[3] == 0
        assert coarse[0] > 0
        assert coarse[2] > 0

    def test_monotonic_mapping(self) -> None:
        """Более высокий F0 → более высокий coarse."""
        f0 = np.array([100.0, 200.0, 400.0, 800.0])
        coarse = f0_to_coarse(f0)
        # Каждый следующий должен быть больше предыдущего
        assert coarse[0] < coarse[1] < coarse[2] < coarse[3]

    def test_range_limits(self) -> None:
        """Значения не выходят за пределы [0, 255]."""
        f0 = np.array([0.0, 1.0, 50.0, 500.0, 1100.0, 5000.0, 10000.0])
        coarse = f0_to_coarse(f0)
        assert coarse.min() >= 0
        assert coarse.max() <= 255

    def test_empty_array(self) -> None:
        """Пустой массив возвращает пустой результат."""
        f0 = np.array([], dtype=np.float64)
        coarse = f0_to_coarse(f0)
        assert len(coarse) == 0
        assert coarse.dtype == np.int32

    def test_custom_range(self) -> None:
        """Пользовательские f0_min и f0_max."""
        f0 = np.array([100.0, 500.0])
        coarse = f0_to_coarse(f0, f0_min=80.0, f0_max=800.0)
        assert coarse.dtype == np.int32
        assert all(1 <= c <= 255 for c in coarse)


class TestPitchMethod:
    """Тесты PitchMethod enum."""

    def test_values(self) -> None:
        """Enum содержит ожидаемые значения."""
        assert PitchMethod.PM.value == "pm"
        assert PitchMethod.HARVEST.value == "harvest"
        assert PitchMethod.RMVPE.value == "rmvpe"
        assert PitchMethod.FCPE.value == "fcpe"

    def test_from_string(self) -> None:
        """Создание из строки."""
        method = PitchMethod("rmvpe")
        assert method == PitchMethod.RMVPE

    def test_invalid_string_raises(self) -> None:
        """Невалидная строка вызывает ValueError."""
        with pytest.raises(ValueError):
            PitchMethod("invalid_method")


class TestPMPitchExtractor:
    """Тесты Parselmouth pitch extractor (CPU)."""

    def test_compute_f0_basic(
        self, sample_audio: np.ndarray, config
    ) -> None:
        """Базовое вычисление F0."""
        extractor = PMPitchExtractor(config)
        f0 = extractor.compute_f0(sample_audio)

        assert isinstance(f0, np.ndarray)
        assert f0.ndim == 1
        assert len(f0) > 0
        assert f0.dtype == np.float64 or f0.dtype == np.float32

    def test_compute_f0_with_p_len(
        self, sample_audio: np.ndarray, config
    ) -> None:
        """F0 с явным p_len."""
        extractor = PMPitchExtractor(config)
        target_len = 100
        f0 = extractor.compute_f0(sample_audio, p_len=target_len)

        assert len(f0) == target_len

    def test_compute_f0_uv_returns_two_arrays(
        self, sample_audio: np.ndarray, config
    ) -> None:
        """compute_f0_uv возвращает (f0, uv)."""
        extractor = PMPitchExtractor(config)
        f0, uv = extractor.compute_f0_uv(sample_audio)

        assert isinstance(f0, np.ndarray)
        assert isinstance(uv, np.ndarray)
        assert f0.shape == uv.shape

    def test_silence_gives_zero_f0(self, silent_audio: np.ndarray, config) -> None:
        """Тихий аудио даёт f0≈0."""
        extractor = PMPitchExtractor(config)
        f0 = extractor.compute_f0(silent_audio)

        # Большинство или все значения должны быть близки к 0
        assert np.mean(f0 < 1.0) > 0.5