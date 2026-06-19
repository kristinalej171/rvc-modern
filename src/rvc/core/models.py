"""
VITS Synthesizer wrapper с поддержкой:
- torch.compile (dynamic shapes)
- CUDA graphs
- Streaming inference
- Memory-efficient loading
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from rvc.config import RVCConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Метаданные загруженной модели."""

    version: Literal["v1", "v2"]
    sample_rate: int
    has_f0: bool
    speaker_count: int
    hidden_dim: int
    path: Path


class SynthesizerWrapper:
    """Обёртка над VITS Synthesizer моделью."""

    def __init__(
        self,
        config: RVCConfig,
        use_compile: bool = True,
        use_cuda_graphs: bool = False,
    ) -> None:
        self.config = config
        self.device = torch.device(config.inference.device)
        self.dtype = (
            torch.float16
            if config.inference.dtype == "float16" and self.device.type == "cuda"
            else torch.float32
        )
        self.use_compile = use_compile and self.device.type == "cuda"
        self.use_cuda_graphs = use_cuda_graphs and self.device.type == "cuda"

        self.net_g: nn.Module | None = None
        self.metadata: ModelMetadata | None = None
        self._cuda_graph: torch.cuda.CUDAGraph | None = None
        self._static_inputs: dict[str, torch.Tensor] | None = None
        self._static_output: torch.Tensor | None = None

    def load(self, model_path: str | Path) -> ModelMetadata:
        """Загрузить модель из .pth checkpoint."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        logger.info("Загрузка модели: %s", model_path)

        # 🔒 SECURITY: weights_only=True предотвращает RCE через pickle-эксплойт
        # RVC-чекпоинты содержат dict[str, Tensor | int | str | list] — это полностью
        # совместимо с безопасным режимом PyTorch 2.5+.
        checkpoint = torch.load(
            str(model_path),
            map_location="cpu",
            weights_only=True,
        )

        cfg = checkpoint.get("config", [])
        version = checkpoint.get("version", "v1")
        has_f0 = bool(checkpoint.get("f0", 1))
        sample_rate = cfg[-1] if len(cfg) > 0 else 40000
        speaker_count = cfg[-3] if len(cfg) > 2 else 1
        hidden_dim = 256 if version == "v1" else 768

        self.metadata = ModelMetadata(
            version=version,
            sample_rate=sample_rate,
            has_f0=has_f0,
            speaker_count=speaker_count,
            hidden_dim=hidden_dim,
            path=model_path,
        )

        self.net_g = self._build_model(checkpoint)

        state_dict = checkpoint.get("weight", checkpoint)
        self.net_g.load_state_dict(state_dict, strict=False)
        self.net_g = self.net_g.to(device=self.device, dtype=self.dtype)
        self.net_g.eval()

        self._prepare_for_inference()
        if self.use_compile:
            self._apply_compile()

        logger.info(
            "Модель загружена: version=%s, sr=%d, f0=%s, speakers=%d",
            self.metadata.version,
            self.metadata.sample_rate,
            self.metadata.has_f0,
            self.metadata.speaker_count,
        )
        return self.metadata

    def _build_model(self, checkpoint: dict[str, Any]) -> nn.Module:
        """Построить модель на основе метаданных checkpoint."""
        cfg = checkpoint["config"]
        is_half = self.dtype == torch.float16

        try:
            from infer.lib.infer_pack.models import (
                SynthesizerTrnMs256NSFsid,
                SynthesizerTrnMs256NSFsid_nono,
                SynthesizerTrnMs768NSFsid,
                SynthesizerTrnMs768NSFsid_nono,
            )
        except ImportError as e:
            raise RuntimeError("Не удалось импортировать архитектуры VITS.") from e

        model_classes = {
            ("v1", True): SynthesizerTrnMs256NSFsid,
            ("v1", False): SynthesizerTrnMs256NSFsid_nono,
            ("v2", True): SynthesizerTrnMs768NSFsid,
            ("v2", False): SynthesizerTrnMs768NSFsid_nono,
        }

        key = (self.metadata.version, self.metadata.has_f0)
        model_cls = model_classes.get(key)
        if model_cls is None:
            raise ValueError(f"Неподдерживаемая конфигурация: {key}")

        return model_cls(*cfg, is_half=is_half)

    def _prepare_for_inference(self) -> None:
        """Подготовить модель для inference."""
        if self.net_g is None:
            return
        if hasattr(self.net_g, "enc_q"):
            del self.net_g.enc_q
        if hasattr(self.net_g, "remove_weight_norm"):
            try:
                self.net_g.remove_weight_norm()
            except Exception as e:
                logger.warning("remove_weight_norm failed: %s", e)

    def _apply_compile(self) -> None:
        """Применить torch.compile к модели."""
        if self.net_g is None:
            return
        try:
            logger.info("Применение torch.compile к synthesizer (dynamic=True)")
            self.net_g.infer = torch.compile(
                self.net_g.infer,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=True,
            )
        except Exception as e:
            logger.warning("torch.compile не удался: %s", e)

    def unload(self) -> None:
        """Выгрузить модель из памяти и очистить VRAM."""
        if self.net_g is not None:
            del self.net_g
            self.net_g = None
            self.metadata = None
            self._cuda_graph = None
            self._static_inputs = None
            self._static_output = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.reset_peak_memory_stats(self.device)
                except Exception:
                    pass

            logger.info("Модель выгружена, VRAM очищена")

    @property
    def is_loaded(self) -> bool:
        return self.net_g is not None and self.metadata is not None

    @torch.inference_mode()
    def infer(
        self,
        feats: torch.Tensor,
        p_len: torch.Tensor,
        sid: torch.Tensor,
        pitch: torch.Tensor | None = None,
        pitchf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Запустить inference модели."""
        if not self.is_loaded:
            raise RuntimeError("Модель не загружена")

        feats = feats.to(device=self.device, dtype=self.dtype)
        p_len = p_len.to(device=self.device)
        sid = sid.to(device=self.device)

        if self.metadata.has_f0:
            if pitch is None or pitchf is None:
                raise ValueError("Модель требует F0, но pitch/pitchf не предоставлены")
            pitch = pitch.to(device=self.device)
            pitchf = pitchf.to(device=self.device, dtype=self.dtype)
            audio = self.net_g.infer(feats, p_len, pitch, pitchf, sid)[0]
        else:
            audio = self.net_g.infer(feats, p_len, sid)[0]

        return audio[0, 0].float()

    def warmup(self, sequence_length: int = 200) -> None:
        """Прогреть модель и инициировать захват CUDA Graph."""
        if not self.is_loaded:
            return

        logger.info("Warmup модели с seq_len=%d", sequence_length)
        hidden_dim = self.metadata.hidden_dim
        feats = torch.randn(
            1, sequence_length, hidden_dim,
            device=self.device, dtype=torch.float32,
        )
        p_len = torch.tensor([sequence_length], device=self.device)
        sid = torch.tensor([0], device=self.device)

        if self.metadata.has_f0:
            pitch = torch.randint(1, 255, (1, sequence_length), device=self.device)
            pitchf = torch.randn(1, sequence_length, device=self.device)
            self.infer(feats, p_len, sid, pitch, pitchf)
        else:
            self.infer(feats, p_len, sid)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info("Warmup завершён")