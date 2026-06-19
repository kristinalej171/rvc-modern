"""
Парсер stdout легаси-скриптов RVC для извлечения структурированных метрик.

Легаси-скрипты RVC (preprocess.py, extract_f0_print.py, train.py) имеют
стабильные форматы вывода, которые можно надёжно парсить через regex.

Этот модуль превращает сырые строки stdout в структурированные TrainingProgress.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class TrainingStage(str, Enum):
    """Этапы тренировки RVC."""
    PREPROCESS = "preprocess"
    EXTRACT_F0 = "extract_f0"
    EXTRACT_FEATURES = "extract_features"
    TRAIN_MODEL = "train_model"
    TRAIN_INDEX = "train_index"


@dataclass
class TrainingMetrics:
    """
    Структурированные метрики тренировки, извлечённые из stdout.
    """
    stage: TrainingStage
    # Прогресс
    current_step: int = 0
    total_steps: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    # Loss метрики (для train_model)
    loss_generator: Optional[float] = None
    loss_discriminator: Optional[float] = None
    loss_mel: Optional[float] = None
    loss_kl: Optional[float] = None
    loss_fm: Optional[float] = None
    # Оптимизатор
    learning_rate: Optional[float] = None
    # Прогресс для этапов предобработки (файлы)
    files_processed: int = 0
    files_total: int = 0
    # F0 extraction
    f0_files_processed: int = 0
    # Время
    eta_seconds: Optional[float] = None
    elapsed_seconds: float = 0.0
    # Исходное сообщение (fallback)
    raw_message: str = ""
    # История loss для графиков
    loss_history_g: list[float] = field(default_factory=list)
    loss_history_d: list[float] = field(default_factory=list)

    @property
    def progress_fraction(self) -> float:
        """Прогресс в долях (0.0 - 1.0)."""
        if self.stage == TrainingStage.TRAIN_MODEL:
            if self.total_epochs > 0:
                return min(1.0, self.current_epoch / self.total_epochs)
            if self.total_steps > 0:
                return min(1.0, self.current_step / self.total_steps)
        elif self.stage in (TrainingStage.PREPROCESS, TrainingStage.EXTRACT_F0):
            if self.files_total > 0:
                return min(1.0, self.files_processed / self.files_total)
        elif self.stage == TrainingStage.TRAIN_INDEX:
            if self.total_steps > 0:
                return min(1.0, self.current_step / self.total_steps)
        return 0.0

    @property
    def eta_formatted(self) -> str:
        """ETA в формате HH:MM:SS."""
        if self.eta_seconds is None:
            return "--:--:--"
        return str(timedelta(seconds=int(self.eta_seconds)))


# ============================================================================
# Regex-паттерны для парсинга stdout легаси-скриптов RVC
# ============================================================================

class TrainingPatterns:
    """
    Коллекция regex-паттернов для парсинга вывода RVC.
    Паттерны основаны на стабильном выводе оригинальных скриптов RVC Project.
    """

    # === PREPROCESS STAGE ===
    # Пример: "100%|██████████| 150/150 [00:12<00:00, 12.50it/s]"
    TQDM_PROGRESS = re.compile(
        r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^\]]*)\]"
    )
    # Пример: "processing 150 files..."
    PREPROCESS_FILES = re.compile(
        r"processing\s+(\d+)\s+files", re.IGNORECASE
    )
    # Пример: "Found 150 audio files(s)"
    FOUND_FILES = re.compile(
        r"[Ff]ound\s+(\d+)\s+(?:audio\s+)?file", re.IGNORECASE
    )

    # === EXTRACT_F0 STAGE ===
    # Пример: "f0/150.wav" или "processing 150/300"
    F0_FILE_PROGRESS = re.compile(
        r"(?:f0/|processing\s+)(\d+)(?:/(\d+))?", re.IGNORECASE
    )
    # Пример: "150/300" в tqdm
    F0_PROGRESS_RATIO = re.compile(r"(\d+)/(\d+)")

    # === EXTRACT_FEATURES STAGE ===
    # Пример: "37-feature-150.npy"
    FEATURE_FILE = re.compile(r"\d+-feature-(\d+)\.npy")

    # === TRAIN_MODEL STAGE === (САМЫЙ ВАЖНЫЙ)
    # Пример оригинального RVC:
    # "Epoch 50, Step 1000: loss_g=0.4521, loss_d=0.1234, loss_mel=23.45, loss_kl=2.34, loss_fm=1.23, lr=0.000100"
    # Или:
    # "[Epoch 50/200][Step 1000/5000] G_loss: 0.4521 D_loss: 0.1234 mel_loss: 23.45 kl_loss: 2.34 fm_loss: 1.23 lr: 0.000100"
    TRAIN_EPOCH_STEP = re.compile(
        r"(?:Epoch|Эпоха)\s+(\d+)(?:/(\d+))?.*?(?:Step|Шаг)\s+(\d+)(?:/(\d+))?",
        re.IGNORECASE,
    )
    TRAIN_LOSS_G = re.compile(
        r"(?:loss_g|G_loss|loss_gen|Generator[_ ]?loss)[:\s]+([\d.]+)",
        re.IGNORECASE,
    )
    TRAIN_LOSS_D = re.compile(
        r"(?:loss_d|D_loss|loss_disc|Discriminator[_ ]?loss)[:\s]+([\d.]+)",
        re.IGNORECASE,
    )
    TRAIN_LOSS_MEL = re.compile(
        r"(?:loss_mel|mel_loss|Mel[_ ]?loss)[:\s]+([\d.]+)",
        re.IGNORECASE,
    )
    TRAIN_LOSS_KL = re.compile(
        r"(?:loss_kl|kl_loss|KL[_ ]?loss)[:\s]+([\d.]+)",
        re.IGNORECASE,
    )
    TRAIN_LOSS_FM = re.compile(
        r"(?:loss_fm|fm_loss|FM[_ ]?loss)[:\s]+([\d.]+)",
        re.IGNORECASE,
    )
    TRAIN_LR = re.compile(
        r"(?:lr|learning[_ ]?rate)[:\s]+([\d.eE+-]+)",
        re.IGNORECASE,
    )
    # Паттерн: "Saved checkpoint at epoch 50" или "Saving model and opt at epoch 50"
    TRAIN_CHECKPOINT = re.compile(
        r"(?:Saved?|Saving|Сохранение).*?(?:checkpoint|model).*?(?:epoch|эпоха)\s+(\d+)",
        re.IGNORECASE,
    )
    # ETA: "ETA: 01:23:45" или "Remaining: 1h 23m"
    TRAIN_ETA = re.compile(
        r"(?:ETA|Remaining|Осталось)[:\s]+(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+)s)?",
        re.IGNORECASE,
    )

    # === TRAIN_INDEX STAGE ===
    # Пример: "Добавлено 8192/50000 vectors"
    INDEX_VECTORS = re.compile(
        r"[Дд]обавлено\s+(\d+)/(\d+)\s+vectors", re.IGNORECASE
    )
    # Пример: "Загружено 50000 vectors"
    INDEX_LOADED = re.compile(
        r"[Зз]агружено\s+(\d+)\s+vectors", re.IGNORECASE
    )
    # KMeans progress
    INDEX_KMEANS = re.compile(
        r"MiniBatchKMeans.*?(\d+)%", re.IGNORECASE
    )


class TrainingOutputParser:
    """
    Парсер stdout легаси-скриптов RVC с буферизацией и извлечением метрик.

    Использование:
        parser = TrainingOutputParser(
            stage=TrainingStage.TRAIN_MODEL,
            total_epochs=200,
        )
        metrics = parser.parse_line(stdout_line)
        if metrics:
            # Обновить UI
            ...
    """

    def __init__(
        self,
        stage: TrainingStage,
        total_epochs: int = 0,
        total_steps: int = 0,
    ) -> None:
        self.stage = stage
        self.metrics = TrainingMetrics(
            stage=stage,
            total_epochs=total_epochs,
            total_steps=total_steps,
        )
        self._start_time = datetime.now()
        self._line_buffer: str = ""
        self._lines_processed: int = 0

    def parse_line(self, line: str) -> Optional[TrainingMetrics]:
        """
        Парсить одну строку stdout. Возвращает обновлённые метрики
        или None, если строка не содержит полезной информации.
        """
        self._lines_processed += 1
        self.metrics.elapsed_seconds = (
            datetime.now() - self._start_time
        ).total_seconds()

        # Буферизация фрагментированных строк
        if not line.endswith("\n"):
            self._line_buffer += line
            return None

        full_line = self._line_buffer + line
        self._line_buffer = ""
        full_line = full_line.strip()

        if not full_line:
            return None

        # Сохраняем raw-сообщение
        self.metrics.raw_message = full_line

        # Диспатчер по этапу
        updated = False
        if self.stage == TrainingStage.PREPROCESS:
            updated = self._parse_preprocess(full_line)
        elif self.stage == TrainingStage.EXTRACT_F0:
            updated = self._parse_extract_f0(full_line)
        elif self.stage == TrainingStage.EXTRACT_FEATURES:
            updated = self._parse_extract_features(full_line)
        elif self.stage == TrainingStage.TRAIN_MODEL:
            updated = self._parse_train_model(full_line)
        elif self.stage == TrainingStage.TRAIN_INDEX:
            updated = self._parse_train_index(full_line)

        return self.metrics if updated else None

    def flush_buffer(self) -> Optional[TrainingMetrics]:
        """Обработать оставшиеся данные в буфере при завершении потока."""
        if self._line_buffer:
            result = self.parse_line(self._line_buffer + "\n")
            self._line_buffer = ""
            return result
        return None

    def _parse_preprocess(self, line: str) -> bool:
        """Парсинг stdout preprocess.py."""
        # Tqdm progress bar
        match = TrainingPatterns.TQDM_PROGRESS.search(line)
        if match:
            current = int(match.group(2))
            total = int(match.group(3))
            self.metrics.files_processed = current
            self.metrics.files_total = total
            return True

        # Found N files
        match = TrainingPatterns.FOUND_FILES.search(line)
        if match:
            self.metrics.files_total = int(match.group(1))
            return True

        return False

    def _parse_extract_f0(self, line: str) -> bool:
        """Парсинг stdout extract_f0_print.py."""
        # Tqdm progress bar
        match = TrainingPatterns.TQDM_PROGRESS.search(line)
        if match:
            self.metrics.f0_files_processed = int(match.group(2))
            self.metrics.files_total = int(match.group(3))
            return True

        # Простой формат "150/300"
        match = TrainingPatterns.F0_PROGRESS_RATIO.search(line)
        if match:
            self.metrics.f0_files_processed = int(match.group(1))
            if match.group(2):
                self.metrics.files_total = int(match.group(2))
            return True

        return False

    def _parse_extract_features(self, line: str) -> bool:
        """Парсинг stdout extract_feature_print.py."""
        # Tqdm progress bar
        match = TrainingPatterns.TQDM_PROGRESS.search(line)
        if match:
            self.metrics.files_processed = int(match.group(2))
            self.metrics.files_total = int(match.group(3))
            return True

        return False

    def _parse_train_model(self, line: str) -> bool:
        """
        Парсинг stdout train.py — самый важный этап.
        Извлекает epoch, step, все виды loss, learning_rate.
        """
        updated = False

        # Epoch + Step
        match = TrainingPatterns.TRAIN_EPOCH_STEP.search(line)
        if match:
            self.metrics.current_epoch = int(match.group(1))
            if match.group(2):
                self.metrics.total_epochs = int(match.group(2))
            self.metrics.current_step = int(match.group(3))
            if match.group(4):
                self.metrics.total_steps = int(match.group(4))
            updated = True

        # Loss Generator
        match = TrainingPatterns.TRAIN_LOSS_G.search(line)
        if match:
            try:
                loss_g = float(match.group(1))
                self.metrics.loss_generator = loss_g
                # Сохраняем в историю (max 500 точек для графиков)
                if len(self.metrics.loss_history_g) > 500:
                    self.metrics.loss_history_g.pop(0)
                self.metrics.loss_history_g.append(loss_g)
                updated = True
            except ValueError:
                pass

        # Loss Discriminator
        match = TrainingPatterns.TRAIN_LOSS_D.search(line)
        if match:
            try:
                self.metrics.loss_discriminator = float(match.group(1))
                if len(self.metrics.loss_history_d) > 500:
                    self.metrics.loss_history_d.pop(0)
                self.metrics.loss_history_d.append(self.metrics.loss_discriminator)
                updated = True
            except ValueError:
                pass

        # Loss Mel
        match = TrainingPatterns.TRAIN_LOSS_MEL.search(line)
        if match:
            try:
                self.metrics.loss_mel = float(match.group(1))
                updated = True
            except ValueError:
                pass

        # Loss KL
        match = TrainingPatterns.TRAIN_LOSS_KL.search(line)
        if match:
            try:
                self.metrics.loss_kl = float(match.group(1))
                updated = True
            except ValueError:
                pass

        # Loss FM
        match = TrainingPatterns.TRAIN_LOSS_FM.search(line)
        if match:
            try:
                self.metrics.loss_fm = float(match.group(1))
                updated = True
            except ValueError:
                pass

        # Learning Rate
        match = TrainingPatterns.TRAIN_LR.search(line)
        if match:
            try:
                self.metrics.learning_rate = float(match.group(1))
                updated = True
            except ValueError:
                pass

        # ETA
        match = TrainingPatterns.TRAIN_ETA.search(line)
        if match:
            try:
                h = int(match.group(1) or 0)
                m = int(match.group(2) or 0)
                s = int(match.group(3) or 0)
                self.metrics.eta_seconds = h * 3600 + m * 60 + s
                updated = True
            except (ValueError, TypeError):
                pass

        # Вычислить ETA, если его нет, но есть прогресс
        if (
            self.metrics.eta_seconds is None
            and self.metrics.current_epoch > 0
            and self.metrics.total_epochs > 0
            and self.metrics.elapsed_seconds > 0
        ):
            progress = self.metrics.current_epoch / self.metrics.total_epochs
            if progress > 0:
                total_time = self.metrics.elapsed_seconds / progress
                self.metrics.eta_seconds = total_time - self.metrics.elapsed_seconds

        return updated

    def _parse_train_index(self, line: str) -> bool:
        """Парсинг stdout скрипта тренировки индекса."""
        # Loaded vectors
        match = TrainingPatterns.INDEX_LOADED.search(line)
        if match:
            self.metrics.total_steps = int(match.group(1))
            return True

        # Added vectors progress
        match = TrainingPatterns.INDEX_VECTORS.search(line)
        if match:
            self.metrics.current_step = int(match.group(1))
            self.metrics.total_steps = int(match.group(2))
            return True

        # KMeans progress
        match = TrainingPatterns.INDEX_KMEANS.search(line)
        if match:
            self.metrics.current_step = int(match.group(1))
            self.metrics.total_steps = 100
            return True

        return False

    @property
    def summary(self) -> str:
        """Краткая сводка текущего состояния."""
        parts = [f"[{self.stage.value}]"]

        if self.stage == TrainingStage.TRAIN_MODEL:
            if self.metrics.current_epoch > 0:
                parts.append(f"Epoch {self.metrics.current_epoch}/{self.metrics.total_epochs}")
            if self.metrics.loss_generator is not None:
                parts.append(f"G_loss={self.metrics.loss_generator:.4f}")
            if self.metrics.loss_discriminator is not None:
                parts.append(f"D_loss={self.metrics.loss_discriminator:.4f}")
            if self.metrics.loss_mel is not None:
                parts.append(f"Mel={self.metrics.loss_mel:.2f}")
            if self.metrics.learning_rate is not None:
                parts.append(f"lr={self.metrics.learning_rate:.2e}")
        elif self.stage in (TrainingStage.PREPROCESS, TrainingStage.EXTRACT_F0):
            if self.metrics.files_total > 0:
                parts.append(f"{self.metrics.files_processed}/{self.metrics.files_total} files")
        elif self.stage == TrainingStage.TRAIN_INDEX:
            if self.metrics.total_steps > 0:
                parts.append(f"{self.metrics.current_step}/{self.metrics.total_steps} vectors")

        if self.metrics.eta_seconds is not None:
            parts.append(f"ETA: {self.metrics.eta_formatted}")

        return " | ".join(parts)