"""Утилиты: безопасный subprocess, работа с путями, скачивание моделей."""
from __future__ import annotations

import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """
    Безопасный запуск внешней команды через subprocess.
    """
    cmd_str = " ".join(shlex.quote(str(c)) for c in cmd)
    logger.debug("Запуск команды: %s", cmd_str)

    try:
        result = subprocess.run(
            [str(c) for c in cmd],
            cwd=cwd,
            timeout=timeout,
            check=check,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            logger.debug("stdout: %s", result.stdout[:500])
        return result
    except subprocess.CalledProcessError as e:
        logger.error("Команда завершилась с ошибкой: %s", cmd_str)
        logger.error("stderr: %s", e.stderr)
        raise
    except subprocess.TimeoutExpired:
        logger.error("Команда превысила таймаут %s: %s", timeout, cmd_str)
        raise


def run_ffmpeg(
    input_path: Path,
    output_path: Path,
    extra_args: Sequence[str] | None = None,
) -> None:
    """Запуск ffmpeg для конвертации аудио."""
    cmd: list[str] = ["ffmpeg", "-y", "-i", str(input_path)]
    if extra_args:
        cmd.extend(str(a) for a in extra_args)
    cmd.append(str(output_path))
    run_command(cmd)


def run_ffprobe(input_path: Path) -> str:
    """Получение информации о медиафайле через ffprobe."""
    result = run_command(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ]
    )
    return result.stdout.strip()


def sanitize_path(path: str | Path) -> Path:
    """
    Очистка и нормализация пути.
    Удаляет пробелы, кавычки, управляющие символы Unicode.
    """
    path_str = str(path).strip()
    # Удаление BOM и directional markers
    for char in ["\u202a", "\u202b", "\u202c", "\u202d", "\u202e", "\ufeff"]:
        path_str = path_str.replace(char, "")
    return Path(path_str).resolve()


def validate_path_within_root(path: str | Path, allowed_root: Path) -> Path:
    """
    🔒 SECURITY: Проверяет, что путь находится внутри разрешённой директории.
    Предотвращает Path Traversal атаки (например, ../../etc/passwd).

    Args:
        path: Путь для валидации.
        allowed_root: Корневая директория, внутри которой должен быть путь.

    Returns:
        Резолвнутый абсолютный путь.

    Raises:
        ValueError: Если путь выходит за пределы allowed_root.
    """
    cleaned = sanitize_path(path)
    allowed_root_resolved = allowed_root.resolve()

    try:
        cleaned.relative_to(allowed_root_resolved)
    except ValueError:
        raise ValueError(
            f"Path traversal detected: '{cleaned}' is outside allowed "
            f"root '{allowed_root_resolved}'"
        )

    return cleaned


def ensure_directory(path: Path) -> Path:
    """Создание директории, если она не существует."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_available_models(weights_dir: Path) -> list[str]:
    """Получение списка доступных .pth моделей."""
    if not weights_dir.exists():
        return []
    return sorted(
        f.name for f in weights_dir.iterdir()
        if f.suffix == ".pth" and f.is_file()
    )


def get_available_indices(index_dir: Path) -> list[str]:
    """Получение списка доступных .index файлов."""
    if not index_dir.exists():
        return []
    return sorted(
        str(f)
        for f in index_dir.rglob("*.index")
        if "trained" not in f.name and f.is_file()
    )


if __name__ == "__main__":
    if "--download-models" in sys.argv:
        from rvc.core.model_downloader import download_all_models
        download_all_models()