"""Кастомные исключения для RVC."""

from typing import Any, Optional


class RVCError(Exception):
    """Базовое исключение для всех ошибок RVC."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ModelNotFoundError(RVCError):
    """Исключение, когда модель не найдена."""

    def __init__(self, model_path: str):
        super().__init__(
            f"Model not found: {model_path}",
            details={"model_path": model_path},
        )


class ModelLoadError(RVCError):
    """Исключение при ошибке загрузки модели."""

    def __init__(self, model_path: str, reason: str):
        super().__init__(
            f"Failed to load model: {model_path}",
            details={"model_path": model_path, "reason": reason},
        )


class InferenceError(RVCError):
    """Исключение при ошибке инференса."""

    def __init__(self, reason: str, context: Optional[dict[str, Any]] = None):
        super().__init__(f"Inference failed: {reason}", details=context or {})


class AudioProcessingError(RVCError):
    """Исключение при ошибке обработки аудио."""

    def __init__(self, file_path: str, reason: str):
        super().__init__(
            f"Audio processing failed: {file_path}",
            details={"file_path": file_path, "reason": reason},
        )


class ConfigurationError(RVCError):
    """Исключение при ошибке конфигурации."""

    def __init__(self, param: str, reason: str):
        super().__init__(
            f"Configuration error for '{param}'",
            details={"parameter": param, "reason": reason},
        )


class DeviceError(RVCError):
    """Исключение при ошибке устройства."""

    def __init__(self, device: str, reason: str):
        super().__init__(
            f"Device error: {device}",
            details={"device": device, "reason": reason},
        )


class ValidationError(RVCError):
    """Исключение при ошибке валидации."""

    def __init__(self, field: str, reason: str):
        super().__init__(
            f"Validation error for '{field}'",
            details={"field": field, "reason": reason},
        )