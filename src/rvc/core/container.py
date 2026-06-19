"""Dependency Injection Container."""
from pathlib import Path

from rvc.config import RVCConfig, get_config
from rvc.core.events import EventBus, event_bus
from rvc.core.pipeline import VCPipeline
from rvc.application.use_cases import VoiceConversionUseCase, ModelManagementUseCase
from rvc.application.training_use_case import TrainingUseCase
from rvc.infrastructure.repositories.file_audio_repository import FileAudioRepository
from rvc.infrastructure.repositories.file_index_repository import FileIndexRepository
from rvc.infrastructure.repositories.file_model_repository import FileModelRepository
from rvc.infrastructure.services.audio_service_impl import AudioServiceImpl
from rvc.infrastructure.services.model_service_impl import ModelServiceImpl
from rvc.infrastructure.services.inference_service_impl import InferenceServiceImpl


class Container:
    """DI Container для управления зависимостями."""

    def __init__(self, config: RVCConfig | None = None):
        self.config = config or get_config()
        self.project_root = Path(__file__).parent.parent.parent.parent

        # Event Bus
        self._event_bus: EventBus = event_bus

        # Repositories
        self._audio_repository: FileAudioRepository | None = None
        self._index_repository: FileIndexRepository | None = None
        self._model_repository: FileModelRepository | None = None

        # Services
        self._audio_service: AudioServiceImpl | None = None
        self._model_service: ModelServiceImpl | None = None
        self._inference_service: InferenceServiceImpl | None = None

        # Pipeline
        self._pipeline: VCPipeline | None = None

        # Use Cases
        self._voice_conversion_use_case: VoiceConversionUseCase | None = None
        self._model_management_use_case: ModelManagementUseCase | None = None
        self._training_use_case: TrainingUseCase | None = None

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def pipeline(self) -> VCPipeline:
        if self._pipeline is None:
            self._pipeline = VCPipeline(
                config=self.config,
                event_bus=self.event_bus,
            )
        return self._pipeline

    @property
    def audio_repository(self) -> FileAudioRepository:
        if self._audio_repository is None:
            self._audio_repository = FileAudioRepository()
        return self._audio_repository

    @property
    def index_repository(self) -> FileIndexRepository:
        if self._index_repository is None:
            self._index_repository = FileIndexRepository(self.config.paths.index_root)
        return self._index_repository

    @property
    def model_repository(self) -> FileModelRepository:
        if self._model_repository is None:
            self._model_repository = FileModelRepository(self.config.paths.weights)
        return self._model_repository

    @property
    def audio_service(self) -> AudioServiceImpl:
        if self._audio_service is None:
            self._audio_service = AudioServiceImpl(self.audio_repository)
        return self._audio_service

    @property
    def model_service(self) -> ModelServiceImpl:
        if self._model_service is None:
            self._model_service = ModelServiceImpl(
                config=self.config,
                model_repository=self.model_repository,
                event_bus=self.event_bus,
            )
        return self._model_service

    @property
    def inference_service(self) -> InferenceServiceImpl:
        if self._inference_service is None:
            self._inference_service = InferenceServiceImpl(
                config=self.config,
                model_service=self.model_service,
                audio_repository=self.audio_repository,
                index_repository=self.index_repository,
                event_bus=self.event_bus,
                pipeline=self.pipeline,
            )
        return self._inference_service

    @property
    def voice_conversion_use_case(self) -> VoiceConversionUseCase:
        if self._voice_conversion_use_case is None:
            # 🔒 SECURITY: передаём project_root для path traversal защиты
            self._voice_conversion_use_case = VoiceConversionUseCase(
                inference_service=self.inference_service,
                audio_service=self.audio_service,
                project_root=self.project_root,
            )
        return self._voice_conversion_use_case

    @property
    def model_management_use_case(self) -> ModelManagementUseCase:
        if self._model_management_use_case is None:
            self._model_management_use_case = ModelManagementUseCase(
                model_service=self.model_service
            )
        return self._model_management_use_case

    @property
    def training_use_case(self) -> TrainingUseCase:
        if self._training_use_case is None:
            self._training_use_case = TrainingUseCase(
                project_root=self.project_root
            )
        return self._training_use_case

    def reset(self):
        """Сбросить все компоненты (для переинициализации)."""
        if self._pipeline is not None:
            self._pipeline.unload()
        self._audio_repository = None
        self._index_repository = None
        self._model_repository = None
        self._audio_service = None
        self._model_service = None
        self._inference_service = None
        self._pipeline = None
        self._voice_conversion_use_case = None
        self._model_management_use_case = None
        self._training_use_case = None


_container_instance: Container | None = None


def init_container(config: RVCConfig | None = None) -> Container:
    global _container_instance
    _container_instance = Container(config)
    return _container_instance


def get_container() -> Container:
    global _container_instance
    if _container_instance is None:
        _container_instance = Container()
    return _container_instance