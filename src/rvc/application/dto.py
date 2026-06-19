"""Data Transfer Objects для Application Layer."""
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Запрос на инференс."""
    input_audio_path: Path
    model_path: Path
    index_path: Optional[Path] = None
    pitch_shift: int = Field(default=0, ge=-24, le=24)
    index_rate: float = Field(default=0.75, ge=0.0, le=1.0)
    rms_mix_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    protect: float = Field(default=0.33, ge=0.0, le=0.5)
    filter_radius: int = Field(default=3, ge=0, le=7)
    resample_sr: int = Field(default=0, ge=0)
    f0_method: str = Field(default="rmvpe")
    output_format: Literal["wav", "mp3", "flac", "m4a"] = Field(
        default="wav", description="Формат выходного аудио"
    )


class InferenceResponse(BaseModel):
    """Ответ инференса."""
    output_audio_path: Path
    duration: float
    sample_rate: int
    processing_time: float
    model_name: str


class TrainingConfig(BaseModel):
    """Конфигурация тренировки модели."""
    exp_name: str = Field(..., description="Название эксперимента")
    trainset_dir: Path = Field(..., description="Путь к директории с обучающими данными")
    sr: str = Field(default="40k", description="Частота дискретизации (32k, 40k, 48k)")
    version: str = Field(default="v2", description="Версия модели (v1, v2)")
    if_f0: bool = Field(default=True, description="Использовать F0 (для пения)")
    np_cpu: int = Field(default=4, ge=1, le=16, description="Количество CPU процессов")
    gpus: str = Field(default="0", description="ID GPU через дефис")
    batch_size: int = Field(default=6, ge=1, le=32, description="Размер батча")
    total_epoch: int = Field(default=200, ge=10, le=1000, description="Всего эпох")
    save_epoch: int = Field(default=10, ge=1, le=100, description="Сохранять каждые N эпох")
    f0_method: str = Field(default="rmvpe", description="Метод извлечения F0")
    if_save_latest: bool = Field(default=True, description="Сохранять только последний checkpoint")
    if_cache_gpu: bool = Field(default=False, description="Кэшировать данные в GPU")
    if_save_every_weights: bool = Field(default=False, description="Сохранять веса каждую эпоху")
    pretrained_G: Optional[str] = Field(default=None, description="Путь к предобученному Generator")
    pretrained_D: Optional[str] = Field(default=None, description="Путь к предобученному Discriminator")
    index_algorithm: str = Field(default="auto", description="Алгоритм индексации")

    class Config:
        arbitrary_types_allowed = True


class TrainingProgress(BaseModel):
    """
    Прогресс тренировки со структурированными метриками.
    Используется для обновления UI и построения live-графиков.
    """
    # Базовые поля
    stage: str
    current: int
    total: int
    message: str
    is_complete: bool = False
    has_error: bool = False

    # === СТРУКТУРИРОВАННЫЕ МЕТРИКИ (из парсера stdout) ===
    # Этап тренировки (для диспетчеризации в UI)
    stage_type: Optional[str] = Field(
        default=None,
        description="Тип этапа: preprocess, extract_f0, extract_features, train_model, train_index"
    )

    # Прогресс (epoch/step)
    epoch: Optional[int] = Field(default=None, description="Текущая эпоха")
    total_epochs: Optional[int] = Field(default=None, description="Всего эпох")
    step: Optional[int] = Field(default=None, description="Текущий шаг")
    total_steps: Optional[int] = Field(default=None, description="Всего шагов")

    # Loss метрики (для train_model)
    loss_generator: Optional[float] = Field(default=None, description="Generator loss")
    loss_discriminator: Optional[float] = Field(default=None, description="Discriminator loss")
    loss_mel: Optional[float] = Field(default=None, description="Mel-spectrogram loss")
    loss_kl: Optional[float] = Field(default=None, description="KL divergence loss")
    loss_fm: Optional[float] = Field(default=None, description="Feature matching loss")

    # Оптимизатор
    learning_rate: Optional[float] = Field(default=None, description="Текущий learning rate")

    # Время
    eta_seconds: Optional[float] = Field(default=None, description="ETA в секундах")
    elapsed_seconds: float = Field(default=0.0, description="Прошло секунд с начала")

    # Файлы (для preprocess/f0/extract)
    files_processed: Optional[int] = Field(default=None, description="Обработано файлов")
    files_total: Optional[int] = Field(default=None, description="Всего файлов")

    # История loss для графиков (список значений)
    loss_history_g: list[float] = Field(default_factory=list)
    loss_history_d: list[float] = Field(default_factory=list)

    # Флаг: содержит ли это обновление новые метрики (для оптимизации UI)
    has_metrics_update: bool = Field(default=False)

    # Сводка для логирования
    summary: str = Field(default="", description="Краткая сводка")