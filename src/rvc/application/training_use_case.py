"""Use Case для тренировки моделей с управлением жизненным циклом процессов."""
import asyncio
import atexit
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from typing import AsyncGenerator, Optional

from rvc.application.dto import TrainingConfig, TrainingProgress
from rvc.application.training_parser import TrainingOutputParser, TrainingStage
from rvc.core.exceptions import RVCError

logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    """Менеджер фоновых задач с PID-трекингом."""

    _instance: Optional["BackgroundTaskManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "BackgroundTaskManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._active_processes: dict[str, asyncio.subprocess.Process] = {}
                cls._instance._process_lock = threading.Lock()
                cls._instance._atexit_registered = False
        return cls._instance

    def register_process(self, task_id: str, process: asyncio.subprocess.Process) -> None:
        with self._process_lock:
            self._active_processes[task_id] = process
            self._ensure_atexit_registered()

    def unregister_process(self, task_id: str) -> None:
        with self._process_lock:
            self._active_processes.pop(task_id, None)

    def _ensure_atexit_registered(self) -> None:
        if not self._atexit_registered:
            atexit.register(self._atexit_cleanup)
            self._atexit_registered = True

    def _atexit_cleanup(self) -> None:
        with self._process_lock:
            active = dict(self._active_processes)
        for task_id, process in active.items():
            if process.returncode is None:
                try:
                    os.kill(process.pid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass

    async def terminate_process(self, task_id: str, grace_period: float = 5.0) -> bool:
        with self._process_lock:
            process = self._active_processes.get(task_id)
        if process is None:
            return False
        try:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=grace_period)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
            self.unregister_process(task_id)
            return True
        except ProcessLookupError:
            self.unregister_process(task_id)
            return True
        except Exception:
            return False

    async def terminate_all(self) -> int:
        with self._process_lock:
            task_ids = list(self._active_processes.keys())
        terminated = sum(1 for tid in task_ids if await self.terminate_process(tid))
        return terminated

    def get_active_count(self) -> int:
        with self._process_lock:
            return len(self._active_processes)


_task_manager = BackgroundTaskManager()


class TrainingUseCase:
    """Use Case для оркестрации процесса тренировки."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.python_executable = sys.executable
        self._current_task_id: Optional[str] = None
        self._current_process: Optional[asyncio.subprocess.Process] = None
        self._task_manager = _task_manager
        self._task_counter: int = 0

    def _generate_task_id(self) -> str:
        self._task_counter += 1
        return f"training_{self._task_counter}_{os.getpid()}"

    def _ensure_directories(self, config: TrainingConfig) -> None:
        """Создает все подпапки, которые ожидают легаси-скрипты."""
        exp_dir = self.project_root / "logs" / config.exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        subdirs = ["0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf"]
        subdirs.append("3_feature256" if config.version == "v1" else "3_feature768")
        for subdir in subdirs:
            (exp_dir / subdir).mkdir(parents=True, exist_ok=True)

        logger.info("Директории для эксперимента '%s' созданы.", config.exp_name)

    async def _run_subprocess_with_metrics(
        self,
        command: list[str],
        stage: TrainingStage,
        config: TrainingConfig,
    ) -> AsyncGenerator[TrainingProgress, None]:
        logger.info("Запуск этапа %s: %s", stage.value, " ".join(command))

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root) + os.pathsep + env.get("PYTHONPATH", "")

        task_id = self._generate_task_id()
        total_epochs = config.total_epoch if stage == TrainingStage.TRAIN_MODEL else 0

        parser = TrainingOutputParser(stage=stage, total_epochs=total_epochs)

        yield TrainingProgress(
            stage=stage.value,
            stage_type=stage.value,
            current=0,
            total=total_epochs or 100,
            message=f"Начало этапа: {stage.value}",
            epoch=0,
            total_epochs=total_epochs if total_epochs > 0 else None,
        )

        try:
            self._current_process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self.project_root),
                env=env,
            )
            self._current_task_id = task_id
            self._task_manager.register_process(task_id, self._current_process)

            while True:
                if self._current_process.stdout is None:
                    break
                line_bytes = await self._current_process.stdout.readline()
                if not line_bytes:
                    break
                try:
                    line = line_bytes.decode("utf-8", errors="replace")
                except UnicodeDecodeError:
                    line = line_bytes.decode("latin-1", errors="replace")

                updated_metrics = parser.parse_line(line)
                if updated_metrics is not None:
                    yield TrainingProgress(
                        stage=stage.value,
                        stage_type=stage.value,
                        current=updated_metrics.current_epoch
                            or updated_metrics.files_processed
                            or updated_metrics.current_step,
                        total=updated_metrics.total_epochs
                            or updated_metrics.files_total
                            or updated_metrics.total_steps,
                        message=parser.summary,
                        summary=parser.summary,
                        epoch=updated_metrics.current_epoch if updated_metrics.current_epoch > 0 else None,
                        total_epochs=updated_metrics.total_epochs if updated_metrics.total_epochs > 0 else None,
                        step=updated_metrics.current_step if updated_metrics.current_step > 0 else None,
                        total_steps=updated_metrics.total_steps if updated_metrics.total_steps > 0 else None,
                        loss_generator=updated_metrics.loss_generator,
                        loss_discriminator=updated_metrics.loss_discriminator,
                        loss_mel=updated_metrics.loss_mel,
                        loss_kl=updated_metrics.loss_kl,
                        loss_fm=updated_metrics.loss_fm,
                        learning_rate=updated_metrics.learning_rate,
                        eta_seconds=updated_metrics.eta_seconds,
                        elapsed_seconds=updated_metrics.elapsed_seconds,
                        files_processed=updated_metrics.files_processed if updated_metrics.files_processed > 0 else None,
                        files_total=updated_metrics.files_total if updated_metrics.files_total > 0 else None,
                        loss_history_g=list(updated_metrics.loss_history_g),
                        loss_history_d=list(updated_metrics.loss_history_d),
                        has_metrics_update=True,
                    )
                else:
                    raw_line = line.strip()
                    if raw_line:
                        yield TrainingProgress(
                            stage=stage.value,
                            stage_type=stage.value,
                            current=0,
                            total=0,
                            message=raw_line,
                            summary=raw_line,
                        )

            parser.flush_buffer()

            await self._current_process.wait()
            if self._current_process.returncode != 0:
                error_msg = (
                    f"[{stage.value}] Процесс завершился с кодом "
                    f"{self._current_process.returncode}"
                )
                yield TrainingProgress(
                    stage=stage.value, stage_type=stage.value,
                    current=0, total=0, message=error_msg, has_error=True,
                )
                raise RVCError(error_msg)

            yield TrainingProgress(
                stage=stage.value,
                stage_type=stage.value,
                current=parser.metrics.total_epochs
                    or parser.metrics.files_total
                    or parser.metrics.total_steps,
                total=parser.metrics.total_epochs
                    or parser.metrics.files_total
                    or parser.metrics.total_steps,
                message=f"Этап {stage.value} завершён",
                summary=parser.summary,
            )

        except asyncio.CancelledError:
            if self._current_process and self._current_process.returncode is None:
                self._current_process.terminate()
                try:
                    await asyncio.wait_for(self._current_process.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    self._current_process.kill()
                    await self._current_process.wait()
            yield TrainingProgress(
                stage=stage.value, stage_type=stage.value,
                current=0, total=0,
                message=f"Этап {stage.value} отменён", has_error=True,
            )
            raise
        except Exception as e:
            error_msg = f"[{stage.value}] Критическая ошибка: {str(e)}"
            yield TrainingProgress(
                stage=stage.value, stage_type=stage.value,
                current=0, total=0, message=error_msg, has_error=True,
            )
            raise RVCError(error_msg)
        finally:
            self._task_manager.unregister_process(task_id)
            self._current_process = None
            self._current_task_id = None

    async def preprocess_data(
        self, config: TrainingConfig
    ) -> AsyncGenerator[TrainingProgress, None]:
        self._ensure_directories(config)
        sr_int = int(config.sr.replace("k", "000"))
        exp_dir = str(self.project_root / "logs" / config.exp_name)
        command = [
            self.python_executable,
            str(self.project_root / "infer" / "modules" / "train" / "preprocess.py"),
            str(config.trainset_dir), str(sr_int), str(config.np_cpu),
            exp_dir, "False", "3.0",
        ]
        async for progress in self._run_subprocess_with_metrics(
            command, TrainingStage.PREPROCESS, config
        ):
            yield progress

    async def extract_f0(
        self, config: TrainingConfig
    ) -> AsyncGenerator[TrainingProgress, None]:
        exp_dir = str(self.project_root / "logs" / config.exp_name)
        command = [
            self.python_executable,
            str(self.project_root / "infer" / "modules" / "train" / "extract" / "extract_f0_print.py"),
            exp_dir, str(config.np_cpu), config.f0_method,
        ]
        async for progress in self._run_subprocess_with_metrics(
            command, TrainingStage.EXTRACT_F0, config
        ):
            yield progress

    async def extract_features(
        self, config: TrainingConfig
    ) -> AsyncGenerator[TrainingProgress, None]:
        device = f"cuda:{config.gpus.split('-')[0]}" if config.gpus else "cpu"
        exp_dir = str(self.project_root / "logs" / config.exp_name)
        command = [
            self.python_executable,
            str(self.project_root / "infer" / "modules" / "train" / "extract_feature_print.py"),
            device, "1", "0", exp_dir, config.version,
        ]
        async for progress in self._run_subprocess_with_metrics(
            command, TrainingStage.EXTRACT_FEATURES, config
        ):
            yield progress

    async def train_model(
        self, config: TrainingConfig
    ) -> AsyncGenerator[TrainingProgress, None]:
        command = [
            self.python_executable,
            str(self.project_root / "infer" / "modules" / "train" / "train.py"),
            "-e", config.exp_name, "-sr", config.sr,
            "-f0", "1" if config.if_f0 else "0",
            "-bs", str(config.batch_size), "-g", config.gpus,
            "-te", str(config.total_epoch), "-se", str(config.save_epoch),
            "-v", config.version,
            "-l", "1" if config.if_save_latest else "0",
            "-c", "1" if config.if_cache_gpu else "0",
            "-sw", "1" if config.if_save_every_weights else "0",
        ]
        if config.pretrained_G:
            command.extend(["-pg", config.pretrained_G])
        if config.pretrained_D:
            command.extend(["-pd", config.pretrained_D])

        async for progress in self._run_subprocess_with_metrics(
            command, TrainingStage.TRAIN_MODEL, config
        ):
            yield progress

    async def train_index(
        self, config: TrainingConfig
    ) -> AsyncGenerator[TrainingProgress, None]:
        """
        Построение FAISS индекса из извлечённых features.

        🔒 SECURITY FIX: Ранее использовался inline Python-скрипт в f-string,
        что было уязвимо к code injection через config.exp_name (содержащий
        одинарную кавычку). Теперь используется отдельный скрипт
        scripts/train_index_script.py, принимающий аргументы через argparse —
        shell-интерполяция полностью исключена.
        """
        exp_dir = self.project_root / "logs" / config.exp_name
        command = [
            self.python_executable,
            str(self.project_root / "scripts" / "train_index_script.py"),
            "--exp-dir", str(exp_dir),
            "--version", config.version,
            "--n-cpu", str(config.np_cpu),
        ]
        async for progress in self._run_subprocess_with_metrics(
            command, TrainingStage.TRAIN_INDEX, config
        ):
            yield progress

    async def train_full(
        self, config: TrainingConfig
    ) -> AsyncGenerator[TrainingProgress, None]:
        stages = [
            ("Предобработка данных", self.preprocess_data),
            ("Извлечение F0", self.extract_f0),
            ("Извлечение features", self.extract_features),
            ("Тренировка модели", self.train_model),
            ("Тренировка индекса", self.train_index),
        ]
        total_stages = len(stages)

        try:
            for stage_idx, (stage_name, stage_func) in enumerate(stages, 1):
                yield TrainingProgress(
                    stage="Общий прогресс",
                    stage_type="overall",
                    current=stage_idx - 1,
                    total=total_stages,
                    message=f"Этап {stage_idx}/{total_stages}: {stage_name}",
                )
                async for progress in stage_func(config):
                    yield progress

            # ИСПРАВЛЕНО: is_complete=True ТОЛЬКО здесь, для всего пайплайна
            yield TrainingProgress(
                stage="Завершено",
                stage_type="overall",
                current=total_stages,
                total=total_stages,
                message="Тренировка успешно завершена!",
                is_complete=True,
            )
        except asyncio.CancelledError:
            yield TrainingProgress(
                stage="Отменено", stage_type="overall",
                current=0, total=total_stages,
                message="Отменено пользователем", has_error=True,
            )
            raise
        except Exception as e:
            yield TrainingProgress(
                stage="Ошибка", stage_type="overall",
                current=0, total=total_stages,
                message=f"Ошибка: {str(e)}", has_error=True,
            )

    async def cancel_training(self) -> bool:
        if self._current_task_id:
            return await self._task_manager.terminate_process(self._current_task_id)
        return False

    async def kill_all_training_processes(self) -> int:
        return await self._task_manager.terminate_all()

    def get_active_training_count(self) -> int:
        return self._task_manager.get_active_count()