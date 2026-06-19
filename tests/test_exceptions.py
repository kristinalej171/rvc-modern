"""
Тесты модуля rvc.core.exceptions.
Покрывают:
- RVCError (базовое исключение)
- Все специфичные исключения
"""
from __future__ import annotations

import pytest

from rvc.core.exceptions import (
    AudioProcessingError,
    ConfigurationError,
    DeviceError,
    InferenceError,
    ModelLoadError,
    ModelNotFoundError,
    RVCError,
    ValidationError,
)


class TestRVCError:
    """Тесты базового RVCError."""

    def test_basic_message(self) -> None:
        """Базовое сообщение."""
        err = RVCError("Test error")
        assert str(err) == "Test error"
        assert err.message == "Test error"
        assert err.details == {}

    def test_with_details(self) -> None:
        """Сообщение с details."""
        err = RVCError("Test", details={"key": "value"})
        assert err.details["key"] == "value"
        assert "Details" in str(err)

    def test_is_exception(self) -> None:
        """Является Exception."""
        err = RVCError("Test")
        assert isinstance(err, Exception)

    def test_can_be_caught(self) -> None:
        """Может быть пойман через try/except."""
        with pytest.raises(RVCError):
            raise RVCError("Test error")


class TestSpecificExceptions:
    """Тесты специфичных исключений."""

    def test_model_not_found_error(self) -> None:
        """ModelNotFoundError."""
        err = ModelNotFoundError("/path/to/model.pth")
        assert "/path/to/model.pth" in err.message
        assert err.details["model_path"] == "/path/to/model.pth"
        assert isinstance(err, RVCError)

    def test_model_load_error(self) -> None:
        """ModelLoadError."""
        err = ModelLoadError("/path/to/model.pth", "Corrupted checkpoint")
        assert "/path/to/model.pth" in err.message
        assert err.details["model_path"] == "/path/to/model.pth"
        assert err.details["reason"] == "Corrupted checkpoint"

    def test_inference_error(self) -> None:
        """InferenceError."""
        err = InferenceError("CUDA OOM", context={"batch_size": 32})
        assert "CUDA OOM" in err.message
        assert err.details["batch_size"] == 32

    def test_audio_processing_error(self) -> None:
        """AudioProcessingError."""
        err = AudioProcessingError("/audio.wav", "Invalid format")
        assert "/audio.wav" in err.message
        assert err.details["file_path"] == "/audio.wav"
        assert err.details["reason"] == "Invalid format"

    def test_configuration_error(self) -> None:
        """ConfigurationError."""
        err = ConfigurationError("batch_size", "Must be > 0")
        assert "batch_size" in err.message
        assert err.details["parameter"] == "batch_size"
        assert err.details["reason"] == "Must be > 0"

    def test_device_error(self) -> None:
        """DeviceError."""
        err = DeviceError("cuda:0", "Out of memory")
        assert "cuda:0" in err.message
        assert err.details["device"] == "cuda:0"
        assert err.details["reason"] == "Out of memory"

    def test_validation_error(self) -> None:
        """ValidationError."""
        err = ValidationError("pitch_shift", "Must be in [-24, 24]")
        assert "pitch_shift" in err.message
        assert err.details["field"] == "pitch_shift"
        assert err.details["reason"] == "Must be in [-24, 24]"


class TestExceptionHierarchy:
    """Тесты иерархии исключений."""

    def test_all_inherit_from_rvc_error(self) -> None:
        """Все специфичные исключения наследуют RVCError."""
        exceptions = [
            ModelNotFoundError("x"),
            ModelLoadError("x", "y"),
            InferenceError("x"),
            AudioProcessingError("x", "y"),
            ConfigurationError("x", "y"),
            DeviceError("x", "y"),
            ValidationError("x", "y"),
        ]

        for exc in exceptions:
            assert isinstance(exc, RVCError)
            assert isinstance(exc, Exception)

    def test_can_catch_specific_via_rvc_error(self) -> None:
        """Специфичные исключения ловятся через RVCError."""
        with pytest.raises(RVCError):
            raise ModelNotFoundError("x")

        with pytest.raises(RVCError):
            raise InferenceError("x")