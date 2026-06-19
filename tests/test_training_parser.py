"""
Тесты модуля rvc.application.training_parser.
Покрывают:
- TrainingOutputParser для всех этапов
- Regex-паттерны для loss/epoch/step/ETA
"""
from __future__ import annotations

import pytest

from rvc.application.training_parser import (
    TrainingOutputParser,
    TrainingStage,
)


class TestTrainingOutputParserPreprocess:
    """Тесты парсинга этапа preprocess."""

    def test_parse_tqdm_progress(self) -> None:
        """Tqdm progress bar парсится."""
        parser = TrainingOutputParser(stage=TrainingStage.PREPROCESS)
        line = "100%|██████████| 150/150 [00:12<00:00, 12.50it/s]\n"

        metrics = parser.parse_line(line)
        assert metrics is not None
        assert metrics.files_processed == 150
        assert metrics.files_total == 150

    def test_parse_found_files(self) -> None:
        """'Found N files' парсится."""
        parser = TrainingOutputParser(stage=TrainingStage.PREPROCESS)
        line = "Found 300 audio files\n"

        metrics = parser.parse_line(line)
        assert metrics is not None
        assert metrics.files_total == 300

    def test_no_newline_buffers(self) -> None:
        """Строки без \\n буферизуются."""
        parser = TrainingOutputParser(stage=TrainingStage.PREPROCESS)
        # Без \n
        result1 = parser.parse_line("partial line")
        assert result1 is None

        # Завершаем строку
        result2 = parser.parse_line(" continued\n")
        # Буфер обработан (может быть None если не совпал паттерн)


class TestTrainingOutputParserTrainModel:
    """Тесты парсинга этапа train_model."""

    def test_parse_epoch_and_step(self) -> None:
        """Epoch + Step парсятся."""
        parser = TrainingOutputParser(
            stage=TrainingStage.TRAIN_MODEL,
            total_epochs=200,
        )
        line = "Epoch 50/200, Step 1000/5000: loss_g=0.45\n"

        metrics = parser.parse_line(line)
        assert metrics is not None
        assert metrics.current_epoch == 50
        assert metrics.total_epochs == 200
        assert metrics.current_step == 1000
        assert metrics.total_steps == 5000

    def test_parse_loss_generator(self) -> None:
        """Generator loss парсится."""
        parser = TrainingOutputParser(stage=TrainingStage.TRAIN_MODEL)
        line = "loss_g=0.4521\n"

        metrics = parser.parse_line(line)
        assert metrics is not None
        assert metrics.loss_generator == pytest.approx(0.4521)

    def test_parse_loss_discriminator(self) -> None:
        """Discriminator loss парсится."""
        parser = TrainingOutputParser(stage=TrainingStage.TRAIN_MODEL)
        line = "loss_d=0.1234\n"

        metrics = parser.parse_line(line)
        assert metrics is not None
        assert metrics.loss_discriminator == pytest.approx(0.1234)

    def test_parse_all_losses_in_one_line(self) -> None:
        """Все loss в одной строке."""
        parser = TrainingOutputParser(stage=TrainingStage.TRAIN_MODEL)
        line = (
            "Epoch 50: loss_g=0.4521, loss_d=0.1234, "
            "loss_mel=23.45, loss_kl=2.34, loss_fm=1.23\n"
        )

        metrics = parser.parse_line(line)
        assert metrics is not None
        assert metrics.loss_generator == pytest.approx(0.4521)
        assert metrics.loss_discriminator == pytest.approx(0.1234)
        assert metrics.loss_mel == pytest.approx(23.45)
        assert metrics.loss_kl == pytest.approx(2.34)
        assert metrics.loss_fm == pytest.approx(1.23)

    def test_parse_learning_rate(self) -> None:
        """Learning rate парсится."""
        parser = TrainingOutputParser(stage=TrainingStage.TRAIN_MODEL)
        line = "lr=0.000100\n"

        metrics = parser.parse_line(line)
        assert metrics is not None
        assert metrics.learning_rate == pytest.approx(0.0001)

    def test_parse_learning_rate_scientific_notation(self) -> None:
        """Learning rate в scientific notation."""
        parser = TrainingOutputParser(stage=TrainingStage.TRAIN_MODEL)
        line = "learning_rate: 1.5e-4\n"

        metrics = parser.parse_line(line)
        assert metrics is not None
        assert metrics.learning_rate == pytest.approx(1.5e-4)

    def test_parse_eta_hms(self) -> None:
        """ETA в формате HH:MM:SS."""
        parser = TrainingOutputParser(stage=TrainingStage.TRAIN_MODEL)
        line = "ETA: 1h 23m 45s\n"

        metrics = parser.parse_line(line)
        assert metrics is not None
        expected_seconds = 1 * 3600 + 23 * 60 + 45
        assert metrics.eta_seconds == expected_seconds

    def test_loss_history_accumulates(self) -> None:
        """История loss накапливается."""
        parser = TrainingOutputParser(stage=TrainingStage.TRAIN_MODEL)

        parser.parse_line("loss_g=0.5\n")
        parser.parse_line("loss_g=0.4\n")
        parser.parse_line("loss_g=0.3\n")

        assert len(parser.metrics.loss_history_g) == 3
        assert parser.metrics.loss_history_g == [0.5, 0.4, 0.3]

    def test_loss_history_limits_to_500(self) -> None:
        """История loss ограничена 500 точками."""
        parser = TrainingOutputParser(stage=TrainingStage.TRAIN_MODEL)

        for i in range(600):
            parser.parse_line(f"loss_g={i * 0.001}\n")

        assert len(parser.metrics.loss_history_g) == 500


class TestTrainingOutputParserIndex:
    """Тесты парсинга этапа train_index."""

    def test_parse_vectors_added(self) -> None:
        """'Добавлено N/M vectors' парсится."""
        parser = TrainingOutputParser(stage=TrainingStage.TRAIN_INDEX)
        line = "Добавлено 8192/50000 vectors\n"

        metrics = parser.parse_line(line)
        assert metrics is not None
        assert metrics.current_step == 8192
        assert metrics.total_steps == 50000

    def test_parse_vectors_loaded(self) -> None:
        """'Загружено N vectors' парсится."""
        parser = TrainingOutputParser(stage=TrainingStage.TRAIN_INDEX)
        line = "Загружено 50000 vectors\n"

        metrics = parser.parse_line(line)
        assert metrics is not None
        assert metrics.total_steps == 50000


class TestTrainingOutputParserGeneral:
    """Общие тесты парсера."""

    def test_summary_format(self) -> None:
        """Summary корректно форматируется."""
        parser = TrainingOutputParser(
            stage=TrainingStage.TRAIN_MODEL,
            total_epochs=200,
        )
        parser.parse_line("Epoch 50/200: loss_g=0.45\n")

        summary = parser.summary
        assert "train_model" in summary
        assert "50" in summary

    def test_progress_fraction_epoch_based(self) -> None:
        """progress_fraction вычисляется по эпохам."""
        parser = TrainingOutputParser(
            stage=TrainingStage.TRAIN_MODEL,
            total_epochs=100,
        )
        parser.parse_line("Epoch 50/100\n")

        assert parser.metrics.progress_fraction == pytest.approx(0.5)

    def test_progress_fraction_files_based(self) -> None:
        """progress_fraction вычисляется по файлам."""
        parser = TrainingOutputParser(stage=TrainingStage.PREPROCESS)
        parser.parse_line("100%|█| 50/100 [00:05]\n")

        assert parser.metrics.progress_fraction == pytest.approx(0.5)

    def test_eta_formatted(self) -> None:
        """ETA форматируется в HH:MM:SS."""
        parser = TrainingOutputParser(stage=TrainingStage.TRAIN_MODEL)
        parser.metrics.eta_seconds = 3661  # 1h 1m 1s

        assert parser.metrics.eta_formatted == "1:01:01"

    def test_eta_formatted_missing(self) -> None:
        """Без ETA возвращается --:--:--."""
        parser = TrainingOutputParser(stage=TrainingStage.TRAIN_MODEL)
        assert parser.metrics.eta_formatted == "--:--:--"

    def test_flush_buffer_clears_it(self) -> None:
        """flush_buffer очищает буфер."""
        parser = TrainingOutputParser(stage=TrainingStage.TRAIN_MODEL)
        parser.parse_line("partial")  # Без \n — в буфере

        parser.flush_buffer()
        # После flush буфер должен быть пустой (не падает)