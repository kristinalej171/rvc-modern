"""
Тесты модуля rvc.core.utils.
Покрывают:
- sanitize_path
- ensure_directory
- get_available_models
- get_available_indices
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rvc.core.utils import (
    ensure_directory,
    get_available_indices,
    get_available_models,
    sanitize_path,
)


class TestSanitizePath:
    """Тесты sanitize_path."""

    def test_basic_path(self) -> None:
        """Обычный путь резолвится."""
        result = sanitize_path("/tmp/test.txt")
        assert isinstance(result, Path)
        assert result.name == "test.txt"

    def test_strips_whitespace(self) -> None:
        """Пробелы по краям удаляются."""
        result = sanitize_path("  /tmp/test.txt  ")
        assert result.name == "test.txt"

    def test_removes_bom(self) -> None:
        """BOM (U+FEFF) удаляется."""
        result = sanitize_path("\ufeff/tmp/test.txt")
        assert "\ufeff" not in str(result)
        assert result.name == "test.txt"

    def test_removes_directional_markers(self) -> None:
        """Unicode directional markers (U+202A..U+202E) удаляются."""
        # Атака через Unicode Directional Override
        malicious = "/tmp/\u202etest.txt\u202c"
        result = sanitize_path(malicious)
        for marker in ["\u202a", "\u202b", "\u202c", "\u202d", "\u202e"]:
            assert marker not in str(result)

    def test_returns_absolute_path(self) -> None:
        """Результат всегда абсолютный."""
        result = sanitize_path("relative/path.txt")
        assert result.is_absolute()

    def test_accepts_path_object(self) -> None:
        """Принимает Path объект."""
        result = sanitize_path(Path("/tmp/test.txt"))
        assert isinstance(result, Path)

    def test_handles_empty_string(self) -> None:
        """Пустая строка резолвится в CWD."""
        result = sanitize_path("")
        assert isinstance(result, Path)


class TestEnsureDirectory:
    """Тесты ensure_directory."""

    def test_creates_directory(self, temp_dir: Path) -> None:
        """Создаёт директорию."""
        new_dir = temp_dir / "new" / "nested" / "dir"
        result = ensure_directory(new_dir)
        assert result.exists()
        assert result.is_dir()
        assert result == new_dir

    def test_idempotent(self, temp_dir: Path) -> None:
        """Повторный вызов не падает."""
        new_dir = temp_dir / "existing"
        ensure_directory(new_dir)
        ensure_directory(new_dir)  # Не должно падать
        assert new_dir.exists()


class TestGetAvailableModels:
    """Тесты get_available_models."""

    def test_empty_directory(self, temp_dir: Path) -> None:
        """Пустая директория возвращает пустой список."""
        result = get_available_models(temp_dir)
        assert result == []

    def test_nonexistent_directory(self) -> None:
        """Несуществующая директория возвращает пустой список."""
        result = get_available_models(Path("/nonexistent/path"))
        assert result == []

    def test_finds_pth_files(self, temp_dir: Path) -> None:
        """Находит .pth файлы."""
        (temp_dir / "model1.pth").touch()
        (temp_dir / "model2.pth").touch()
        (temp_dir / "not_model.txt").touch()

        result = get_available_models(temp_dir)
        assert len(result) == 2
        assert "model1.pth" in result
        assert "model2.pth" in result
        assert "not_model.txt" not in result

    def test_sorted_alphabetically(self, temp_dir: Path) -> None:
        """Результат отсортирован."""
        (temp_dir / "z_model.pth").touch()
        (temp_dir / "a_model.pth").touch()
        (temp_dir / "m_model.pth").touch()

        result = get_available_models(temp_dir)
        assert result == ["a_model.pth", "m_model.pth", "z_model.pth"]

    def test_ignores_directories(self, temp_dir: Path) -> None:
        """Игнорирует директории с расширением .pth."""
        (temp_dir / "real_model.pth").touch()
        (temp_dir / "fake_model.pth").mkdir()

        result = get_available_models(temp_dir)
        assert len(result) == 1
        assert result[0] == "real_model.pth"


class TestGetAvailableIndices:
    """Тесты get_available_indices."""

    def test_empty_directory(self, temp_dir: Path) -> None:
        """Пустая директория возвращает пустой список."""
        result = get_available_indices(temp_dir)
        assert result == []

    def test_finds_index_files(self, temp_dir: Path) -> None:
        """Находит .index файлы."""
        (temp_dir / "my_model_IVF256_Flat.index").touch()
        (temp_dir / "another.index").touch()
        (temp_dir / "not_index.txt").touch()

        result = get_available_indices(temp_dir)
        assert len(result) == 2

    def test_excludes_trained_indices(self, temp_dir: Path) -> None:
        """Исключает файлы со словом 'trained' в имени."""
        (temp_dir / "model_added.index").touch()
        (temp_dir / "model_trained.index").touch()

        result = get_available_indices(temp_dir)
        # Только один индекс (trained исключён)
        assert len(result) == 1
        assert "trained" not in result[0]

    def test_recursive_search(self, temp_dir: Path) -> None:
        """Рекурсивный поиск во вложенных директориях."""
        nested = temp_dir / "subdir" / "deep"
        nested.mkdir(parents=True)
        (nested / "nested_model.index").touch()

        result = get_available_indices(temp_dir)
        assert len(result) == 1
        assert "nested_model.index" in result[0]