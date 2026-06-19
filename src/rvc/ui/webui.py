"""Gradio WebUI для RVC с live-метриками тренировки."""
import asyncio
import logging
from pathlib import Path
from typing import Optional

import gradio as gr

# Опциональная зависимость для графиков
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from rvc.application.dto import InferenceRequest, TrainingConfig, TrainingProgress
from rvc.core.container import get_container

logger = logging.getLogger(__name__)


def _build_loss_plot(progress: TrainingProgress) -> Optional[object]:
    """Построить plotly-график loss-кривых из истории."""
    if not PLOTLY_AVAILABLE:
        return None

    if not progress.loss_history_g and not progress.loss_history_d:
        fig = go.Figure()
        fig.add_annotation(
            text="Ожидание данных loss...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#888"),
        )
        fig.update_layout(
            title="Training Loss",
            height=300,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if progress.loss_history_g:
        fig.add_trace(
            go.Scatter(
                y=progress.loss_history_g,
                name="G_Loss",
                line=dict(color="#00d4ff", width=2),
                mode="lines",
            ),
            secondary_y=False,
        )
    if progress.loss_history_d:
        fig.add_trace(
            go.Scatter(
                y=progress.loss_history_d,
                name="D_Loss",
                line=dict(color="#ff6b6b", width=2),
                mode="lines",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=f"Training Loss (Epoch {progress.epoch or 0})",
        xaxis_title="Step",
        height=300,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        showlegend=True,
    )
    fig.update_yaxes(title_text="Generator Loss", secondary_y=False)
    fig.update_yaxes(title_text="Discriminator Loss", secondary_y=True)
    return fig


def create_inference_tab(container) -> None:
    """Создать вкладку инференса."""
    gr.Markdown("## 🎤 Инференс (преобразование голоса)")

    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(
                label="Входное аудио",
                type="filepath",
                sources=["upload", "microphone"],
            )
            model_path = gr.Dropdown(
                label="Модель",
                choices=[str(m.path) for m in container.model_repository.get_all()],
                interactive=True,
            )
            index_path = gr.Dropdown(
                label="Индекс (опционально)",
                choices=[""] + [
                    str(p) for p in container.config.paths.logs.rglob("*.index")
                ],
                interactive=True,
            )

        with gr.Column():
            pitch_shift = gr.Slider(
                minimum=-24, maximum=24, value=0, step=1,
                label="Сдвиг тона (полутоны)",
            )
            index_rate = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.75, step=0.05,
                label="Index Rate",
            )
            rms_mix_rate = gr.Slider(
                minimum=0.0, maximum=1.0, value=1.0, step=0.05,
                label="RMS Mix Rate",
            )
            protect = gr.Slider(
                minimum=0.0, maximum=0.5, value=0.33, step=0.01,
                label="Protect (защита согласных)",
            )

    with gr.Row():
        infer_button = gr.Button("🚀 Преобразовать", variant="primary")
    output_audio = gr.Audio(label="Результат", type="filepath")

    async def run_inference(
        input_audio_path, model_path_str, index_path_str,
        pitch, idx_rate, rms_rate, protect_val,
    ):
        try:
            request = InferenceRequest(
                input_audio_path=Path(input_audio_path),
                model_path=Path(model_path_str),
                index_path=Path(index_path_str) if index_path_str else None,
                pitch_shift=pitch,
                index_rate=idx_rate,
                rms_mix_rate=rms_rate,
                protect=protect_val,
            )
            response = await container.voice_conversion_use_case.convert_single(request)
            return str(response.output_audio_path)
        except Exception as e:
            logger.exception("Ошибка инференса")
            gr.Error(f"Ошибка: {str(e)}")
            return None

    infer_button.click(
        fn=run_inference,
        inputs=[
            input_audio, model_path, index_path,
            pitch_shift, index_rate, rms_mix_rate, protect,
        ],
        outputs=[output_audio],
    )


def create_training_tab(container) -> None:
    """Создать вкладку тренировки с live-метриками и loss-графиками."""
    gr.Markdown("## 🎓 Тренировка модели")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Основные параметры")
            exp_name = gr.Textbox(
                label="Название эксперимента",
                placeholder="my_voice_model",
            )
            trainset_dir = gr.Textbox(
                label="Путь к обучающим данным",
                placeholder="/path/to/dataset",
            )

            with gr.Row():
                sr = gr.Dropdown(
                    choices=["32k", "40k", "48k"], value="40k",
                    label="Частота дискретизации",
                )
                version = gr.Dropdown(
                    choices=["v1", "v2"], value="v2", label="Версия модели",
                )
            if_f0 = gr.Checkbox(label="Использовать F0 (для пения)", value=True)

        with gr.Column():
            gr.Markdown("### Параметры тренировки")
            with gr.Row():
                batch_size = gr.Slider(
                    minimum=1, maximum=32, value=6, step=1, label="Batch Size",
                )
                total_epoch = gr.Slider(
                    minimum=10, maximum=1000, value=200, step=10,
                    label="Всего эпох",
                )

            with gr.Row():
                save_epoch = gr.Slider(
                    minimum=1, maximum=100, value=10, step=1,
                    label="Сохранять каждые N эпох",
                )
                np_cpu = gr.Slider(
                    minimum=1, maximum=16, value=4, step=1,
                    label="CPU процессов",
                )

            gpus = gr.Textbox(
                label="GPU ID", value="0",
                info="ID GPU через дефис (например: 0-1-2)",
            )
            f0_method = gr.Dropdown(
                choices=["pm", "harvest", "crepe", "rmvpe"],
                value="rmvpe", label="Метод извлечения F0",
            )

    with gr.Accordion("Дополнительные параметры", open=False):
        with gr.Row():
            if_save_latest = gr.Checkbox(
                label="Сохранять только последний checkpoint", value=True,
            )
            if_cache_gpu = gr.Checkbox(
                label="Кэшировать данные в GPU", value=False,
            )
            if_save_every_weights = gr.Checkbox(
                label="Сохранять веса каждую эпоху", value=False,
            )

        with gr.Row():
            pretrained_G = gr.Textbox(
                label="Предобученный Generator (опционально)",
                placeholder="assets/pretrained_v2/f0G40k.pth",
            )
            pretrained_D = gr.Textbox(
                label="Предобученный Discriminator (опционально)",
                placeholder="assets/pretrained_v2/f0D40k.pth",
            )

    with gr.Row():
        train_button = gr.Button(
            "🚀 Начать тренировку", variant="primary", size="lg",
        )
        stop_button = gr.Button("⏹ Остановить", variant="stop", size="lg")

    gr.Markdown("### 📊 Метрики тренировки")
    with gr.Row():
        stage_display = gr.Textbox(label="Текущий этап", interactive=False)
        epoch_display = gr.Textbox(
            label="Эпоха", interactive=False, value="-- / --",
        )
        step_display = gr.Textbox(label="Шаг", interactive=False, value="--")

    with gr.Row():
        loss_g_display = gr.Textbox(
            label="Generator Loss", interactive=False, value="--",
        )
        loss_d_display = gr.Textbox(
            label="Discriminator Loss", interactive=False, value="--",
        )
        loss_mel_display = gr.Textbox(
            label="Mel Loss", interactive=False, value="--",
        )
        lr_display = gr.Textbox(
            label="Learning Rate", interactive=False, value="--",
        )
        eta_display = gr.Textbox(
            label="ETA", interactive=False, value="--:--:--",
        )

    progress_bar = gr.Slider(
        minimum=0.0, maximum=1.0, value=0.0,
        label="Прогресс этапа", interactive=False,
    )
    overall_progress = gr.Slider(
        minimum=0.0, maximum=1.0, value=0.0,
        label="Общий прогресс этапов", interactive=False,
    )

    if PLOTLY_AVAILABLE:
        loss_plot = gr.Plot(label="Loss Curves (Live)")
    else:
        loss_plot = None
        gr.Markdown(
            "⚠️ Plotly не установлен. Графики loss недоступны. "
            "Установите: `pip install plotly`"
        )

    with gr.Accordion("Подробные логи", open=False):
        log_output = gr.Textbox(
            label="Логи тренировки",
            lines=15, max_lines=50,
            interactive=False, autoscroll=True,
        )

    # Динамический список outputs
    outputs = [
        stage_display, epoch_display, step_display,
        loss_g_display, loss_d_display, loss_mel_display,
        lr_display, eta_display,
        progress_bar, overall_progress,
    ]
    if loss_plot is not None:
        outputs.append(loss_plot)
    outputs.append(log_output)

    async def start_training(
        exp_name_val, trainset_dir_val, sr_val, version_val, if_f0_val,
        batch_size_val, total_epoch_val, save_epoch_val, np_cpu_val,
        gpus_val, f0_method_val, if_save_latest_val, if_cache_gpu_val,
        if_save_every_weights_val, pretrained_G_val, pretrained_D_val,
    ):
        try:
            if not exp_name_val or not trainset_dir_val:
                gr.Error("Заполните название эксперимента и путь к данным")
                return
            if not Path(trainset_dir_val).exists():
                gr.Error(f"Директория не найдена: {trainset_dir_val}")
                return

            config = TrainingConfig(
                exp_name=exp_name_val,
                trainset_dir=Path(trainset_dir_val),
                sr=sr_val,
                version=version_val,
                if_f0=if_f0_val,
                batch_size=batch_size_val,
                total_epoch=total_epoch_val,
                save_epoch=save_epoch_val,
                np_cpu=np_cpu_val,
                gpus=gpus_val,
                f0_method=f0_method_val,
                if_save_latest=if_save_latest_val,
                if_cache_gpu=if_cache_gpu_val,
                if_save_every_weights=if_save_every_weights_val,
                pretrained_G=pretrained_G_val if pretrained_G_val else None,
                pretrained_D=pretrained_D_val if pretrained_D_val else None,
            )

            all_logs: list[str] = []
            current_plot = None

            async for progress in container.training_use_case.train_full(config):
                if progress.summary:
                    log_entry = progress.summary
                else:
                    log_entry = f"[{progress.stage}] {progress.message}"

                if log_entry not in all_logs[-5:]:
                    all_logs.append(log_entry)

                stage_val = progress.stage
                epoch_val = (
                    f"{progress.epoch or 0} / {progress.total_epochs or '?'}"
                    if progress.epoch else "-- / --"
                )
                step_val = str(progress.step or "--")
                loss_g_val = (
                    f"{progress.loss_generator:.4f}"
                    if progress.loss_generator is not None else "--"
                )
                loss_d_val = (
                    f"{progress.loss_discriminator:.4f}"
                    if progress.loss_discriminator is not None else "--"
                )
                loss_mel_val = (
                    f"{progress.loss_mel:.2f}"
                    if progress.loss_mel is not None else "--"
                )
                lr_val = (
                    f"{progress.learning_rate:.2e}"
                    if progress.learning_rate is not None else "--"
                )

                if progress.eta_seconds is not None:
                    h = int(progress.eta_seconds // 3600)
                    m = int((progress.eta_seconds % 3600) // 60)
                    s = int(progress.eta_seconds % 60)
                    eta_val = f"{h:02d}:{m:02d}:{s:02d}"
                else:
                    eta_val = "--:--:--"

                if progress.total and progress.total > 0:
                    progress_val = progress.current / progress.total
                else:
                    progress_val = 0.0

                if progress.stage_type == "overall":
                    overall_val = progress_val
                    prog_val = 0.0
                else:
                    overall_val = 0.0
                    prog_val = progress_val

                if progress.has_metrics_update and loss_plot is not None:
                    current_plot = _build_loss_plot(progress)

                res = [
                    stage_val, epoch_val, step_val,
                    loss_g_val, loss_d_val, loss_mel_val, lr_val, eta_val,
                    prog_val, overall_val,
                ]
                if loss_plot is not None:
                    res.append(current_plot)
                res.append("\n".join(all_logs[-50:]))
                yield tuple(res)

                if progress.has_error:
                    gr.Error(progress.message)
                    break
                if progress.is_complete:
                    gr.Info("Тренировка успешно завершена!")
                    break

        except Exception as e:
            logger.exception("Ошибка тренировки")
            gr.Error(f"Критическая ошибка: {str(e)}")
            res = [None] * len(outputs)
            res[0] = "Ошибка"
            res[-1] = f"Критическая ошибка: {str(e)}"
            yield tuple(res)

    async def stop_training():
        try:
            cancelled = await container.training_use_case.cancel_training()
            if cancelled:
                gr.Info("Тренировка остановлена")
                res = [
                    "Остановлено", "-- / --", "--", "--", "--", "--", "--",
                    "--:--:--", 0.0, 0.0,
                ]
                if loss_plot is not None:
                    res.append(None)
                res.append("Тренировка остановлена пользователем")
                return tuple(res)
            else:
                gr.Warning("Нет активной тренировки")
                return tuple([None] * len(outputs))
        except Exception as e:
            logger.exception("Ошибка остановки")
            gr.Error(f"Ошибка: {str(e)}")
            return tuple([None] * len(outputs))

    train_button.click(
        fn=start_training,
        inputs=[
            exp_name, trainset_dir, sr, version, if_f0,
            batch_size, total_epoch, save_epoch, np_cpu,
            gpus, f0_method, if_save_latest, if_cache_gpu,
            if_save_every_weights, pretrained_G, pretrained_D,
        ],
        outputs=outputs,
        show_progress="hidden",
    )

    stop_button.click(
        fn=stop_training,
        inputs=[],
        outputs=outputs,
    )


def build_webui(container) -> gr.Blocks:
    """
    Создать Gradio WebUI.

    🔧 BUG FIX: Для Gradio 5.x (зафиксирован в pyproject.toml):
    - theme передаётся в gr.Blocks(...), а НЕ в launch()
    - queue() использует параметр concurrency_count (а не default_concurrency_limit)
    """
    # 🔧 BUG FIX: theme возвращён в gr.Blocks (правильно для Gradio 5.x)
    with gr.Blocks(
        title="RVC - Retrieval-based Voice Conversion",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# 🎙️ RVC - Retrieval-based Voice Conversion")
        gr.Markdown("Современный интерфейс для преобразования голоса")

        with gr.Tabs():
            with gr.Tab("Инференс"):
                create_inference_tab(container)
            with gr.Tab("Тренировка"):
                create_training_tab(container)

        gr.Markdown("---")
        gr.Markdown(
            "### 📚 Документация\n"
            "- **Инференс**: Преобразование голоса с использованием обученной модели\n"
            "- **Тренировка**: Обучение новой модели на ваших данных\n"
            "\n"
            "### ⚙️ Требования\n"
            "- CUDA-совместимая GPU (рекомендуется)\n"
            "- Минимум 8GB VRAM для тренировки\n"
            "- Python 3.11+\n"
        )

    # 🔒 SECURITY: Rate limiting (из Фазы 1)
    # 🔧 BUG FIX: concurrency_count — стабильный API для Gradio 5.x
    # В Gradio 6.x этот параметр был переименован в default_concurrency_limit,
    # но мы зафиксировали версию на 5.x для стабильности.
    demo.queue(
        concurrency_count=2,  # Макс 2 параллельных запроса (защита от OOM)
        max_size=20,           # Макс 20 в очереди
    )

    return demo


def launch_webui(
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
):
    """Запустить WebUI сервер."""
    container = get_container()
    demo = build_webui(container)
    # queue() уже настроен в build_webui()
    # 🔧 BUG FIX: theme убран отсюда (в Gradio 5.x он передаётся в Blocks)
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
    )


if __name__ == "__main__":
    launch_webui()