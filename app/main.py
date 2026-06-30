# -*- coding: utf-8 -*-
"""
FastAPI-сервис для рекомендательной системы Kion.
Эндпоинты: /predict, /health, /model-info, /retrain
Мониторинг: Prometheus метрики.
"""
import json
import logging
import threading
from typing import List, Optional
from contextlib import asynccontextmanager

from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import time

import jinja2
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY

from src.models.predict_model import Predictor
from src.drift.detector import DriftDetector, DriftReportData
from src.drift.report import save_report, generate_html, REPORTS_DIR

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Глобальный предиктор (кешируем загрузку модели)
_predictors: dict = {}

# --- Prometheus метрики ---
REQUEST_COUNT = Counter(
    'recsys_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'recsys_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)
ACTIVE_REQUESTS = Histogram(
    'recsys_active_requests',
    'Number of in-flight requests',
    buckets=(0, 1, 2, 5, 10, 20, 50)
)

PREDICTION_COUNTER = Counter(
    'recsys_predictions_total',
    'Total number of predictions served',
    ['model_type']
)
MODEL_SCORE_GAUGE = Gauge(
    'recsys_model_metric',
    'Метрики модели (MAP, Precision, Recall, Novelty)',
    ['model_type', 'metric']
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружаем предикторы при старте."""
    logger.info("Загрузка моделей...")
    try:
        _predictors['popular'] = Predictor(model_type='popular')
        logger.info("PopularRecommender загружен")
    except Exception as e:
        logger.warning(f"Не удалось загрузить PopularRecommender: {e}")

    try:
        _predictors['tfidf'] = Predictor(model_type='tfidf')
        logger.info("TFIDFRecommender загружен")
    except Exception as e:
        logger.warning(f"Не удалось загрузить TFIDFRecommender: {e}")

    logger.info(f"Сервис запущен. Доступные модели: {list(_predictors.keys())}")

    yield
    _predictors.clear()


app = FastAPI(
    title="Kion RecSys API",
    description="Рекомендательная система для онлайн-кинотеатра Kion",
    version="0.2.0",
    lifespan=lifespan,
)


# --- Prometheus middleware — собираем метрики на каждый запрос ---
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path

    start_time = time.monotonic()
    response = await call_next(request)
    latency = time.monotonic() - start_time

    status = str(response.status_code)
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)

    return response


# --- Схемы запросов/ответов ---

class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="ID пользователя")
    top_k: int = Field(default=10, ge=1, le=100, description="Количество рекомендаций")
    model_type: str = Field(default='popular', pattern='^(popular|tfidf)$')


class RecommendItem(BaseModel):
    item_id: int
    rank: int
    score: Optional[float] = None


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendItem]
    model_type: str


class ModelInfoResponse(BaseModel):
    available_models: List[str]
    loaded_models: List[str]
    version: str


class HealthResponse(BaseModel):
    status: str
    loaded_models: List[str]


# --- Web UI ---

# Монтируем статику
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
template_loader = jinja2.FileSystemLoader(searchpath=Path(__file__).parent / "templates")
template_env = jinja2.Environment(loader=template_loader)


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def index(request: Request):
    """Web UI главная страница."""
    template = template_env.get_template("index.html")
    return HTMLResponse(content=template.render(
        request=request,
        models_available=list(_predictors.keys()),
    ))


# --- Переобучение ---

def _reload_predictors():
    """Перезагружает модели после переобучения."""
    _predictors.clear()
    try:
        _predictors['popular'] = Predictor(model_type='popular')
    except Exception as e:
        logger.error(f"Не удалось загрузить PopularRecommender после переобучения: {e}")
    try:
        _predictors['tfidf'] = Predictor(model_type='tfidf')
    except Exception as e:
        logger.error(f"Не удалось загрузить TFIDFRecommender после переобучения: {e}")
    logger.info(f"Модели перезагружены. Доступны: {list(_predictors.keys())}")


def _export_metrics_to_prometheus(output: str, model_type: str):
    """Парсит stdout обучения и экспортирует метрики в Prometheus."""
    import re
    # Ищем: METRICS:model_type:{'MAP_10': ...}
    match = re.search(rf"METRICS:{model_type}:(\{{.+\}})", output)
    if match:
        import ast
        try:
            metrics_str = match.group(1)
            # Заменяем np.float64(...) на обычные числа
            metrics_str = re.sub(r'np\.float64\((\S+)\)', r'\1', metrics_str)
            metrics_dict = ast.literal_eval(metrics_str)
            for metric_name, metric_val in metrics_dict.items():
                MODEL_SCORE_GAUGE.labels(model_type=model_type, metric=metric_name).set(float(metric_val))
            logger.info(f"Метрики {model_type} экспортированы в Prometheus: {metrics_dict}")
        except Exception as e:
            logger.warning(f"Не удалось распарсить метрики {model_type}: {e}")


def _run_training():
    """Запускает обучение моделей в том же интерпретаторе (без subprocess)."""
    from click.testing import CliRunner
    from src.models.train_model import train as train_command

    project_dir = Path(__file__).resolve().parents[1]
    processed_path = project_dir / 'data' / 'processed' / 'interactions_processed.csv'
    features_path = project_dir / 'data' / 'features'
    models_path = project_dir / 'models'

    logger.info("Запуск переобучения PopularRecommender...")
    runner = CliRunner()
    result = runner.invoke(train_command, [
        str(processed_path),
        str(features_path),
        str(models_path / 'popular.pkl'),
        '--model_type', 'popular'
    ])
    if result.exit_code == 0:
        logger.info("PopularRecommender обучен")
        _export_metrics_to_prometheus(result.output, 'popular')
    else:
        logger.error(f"Ошибка обучения popular: {result.output}")

    logger.info("Запуск переобучения TFIDFRecommender...")
    result = runner.invoke(train_command, [
        str(processed_path),
        str(features_path),
        str(models_path / 'tfidf.pkl'),
        '--model_type', 'tfidf'
    ])
    if result.exit_code == 0:
        logger.info("TFIDFRecommender обучен")
        _export_metrics_to_prometheus(result.output, 'tfidf')
    else:
        logger.error(f"Ошибка обучения tfidf: {result.output}")

    _reload_predictors()
    logger.info("Переобучение завершено")


@app.post("/retrain", tags=["System"])
async def retrain():
    """Запуск переобучения моделей в фоне."""
    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()
    logger.info("Запрос на переобучение — запущено в фоне")
    return {
        "status": "training_started",
        "message": "Переобучение запущено в фоне. Модели обновятся через ~1-2 минуты."
    }


@app.get("/retrain/status", tags=["System"])
async def retrain_status():
    """Статус переобучения (живы ли текущие модели)."""
    return {
        "models": list(_predictors.keys()),
        "ok": len(_predictors) > 0,
    }


# --- Drift detection ---

_last_drift_report: Optional[DriftReportData] = None
_drift_running: bool = False


def _run_drift_check(train_start: str, train_end: str, test_start: str, test_end: str):
    """Запуск дрифт-чека в фоне."""
    global _last_drift_report, _drift_running
    _drift_running = True
    try:
        detector = DriftDetector()
        report = detector.run_full_check(train_start, train_end, test_start, test_end)
        save_report(report)
        _last_drift_report = report
        logger.info(f"Drift check completed: {report.drift_count} drifts found")
    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        raise
    finally:
        _drift_running = False


@app.post("/drift/run", tags=["Drift"])
async def drift_run(
    train_start: str = Query(default="2021-03-13", description="Start of train period"),
    train_end: str = Query(default="2021-06-01", description="End of train period"),
    test_start: str = Query(default="2021-06-01", description="Start of test period"),
    test_end: str = Query(default="2021-08-23", description="End of test period"),
):
    """Запустить дрифт-чек в фоне. Сравнивает два временных периода."""
    if _drift_running:
        return {"status": "busy", "message": "Drift check уже выполняется"}
    thread = threading.Thread(
        target=_run_drift_check,
        args=(train_start, train_end, test_start, test_end),
        daemon=True,
    )
    thread.start()
    return {
        "status": "started",
        "message": f"Drift check запущен: train=[{train_start}, {train_end}), test=[{test_start}, {test_end})",
    }


@app.get("/drift/status", tags=["Drift"])
async def drift_status():
    """Статус последнего дрифт-чека."""
    if _drift_running:
        return {"status": "running", "message": "Drift check выполняется..."}
    if _last_drift_report is None:
        return {"status": "never_run", "message": "Drift check ещё не запускался"}
    return {
        "status": "ok",
        "has_drift": _last_drift_report.has_any_drift,
        "drift_count": _last_drift_report.drift_count,
        "concept_drift": _last_drift_report.concept_drift,
        "timestamp": _last_drift_report.timestamp,
        "train_period": list(_last_drift_report.train_period),
        "test_period": list(_last_drift_report.test_period),
    }


@app.get("/drift/report", tags=["Drift"])
async def drift_report():
    """HTML-отчёт последнего дрифт-чека."""
    if _drift_running:
        return HTMLResponse("<h3>⏳ Drift check выполняется...</h3>")
    if _last_drift_report is None:
        return HTMLResponse("<h3>Drift check ещё не запускался</h3>")
    return HTMLResponse(content=generate_html(_last_drift_report))


@app.get("/drift/data", tags=["Drift"])
async def drift_data():
    """JSON-данные последнего дрифт-чека."""
    if _last_drift_report is None:
        return {"status": "never_run"}
    return _last_drift_report.to_dict()


@app.get("/drift/history", tags=["Drift"])
async def drift_history():
    """Список всех сохранённых отчётов о дрейфе."""
    if not REPORTS_DIR.exists():
        return {"reports": []}
    files = sorted(REPORTS_DIR.glob("drift_report_*.json"), reverse=True)
    reports = []
    for f in files[:20]:
        try:
            data = json.loads(f.read_text())
            reports.append({
                "filename": f.name,
                "timestamp": data.get("timestamp", ""),
                "has_drift": data.get("has_any_drift", False),
                "drift_count": data.get("drift_count", 0),
                "train_period": data.get("train_period", []),
                "test_period": data.get("test_period", []),
            })
        except Exception:
            pass
    return {"reports": reports}


# --- Эндпоинты ---

@app.get("/metrics", tags=["System"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(REGISTRY), media_type="text/plain")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Проверка работоспособности сервиса."""
    return HealthResponse(
        status="ok",
        loaded_models=list(_predictors.keys()),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Информация о доступных и загруженных моделях."""
    return ModelInfoResponse(
        available_models=["popular", "tfidf"],
        loaded_models=list(_predictors.keys()),
        version="0.2.0",
    )


@app.post("/predict/{user_id}", response_model=RecommendResponse, tags=["Recommendations"])
async def predict(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=100),
    model_type: str = Query(default='popular', pattern='^(popular|tfidf)$'),
):
    """
    Получить top-K рекомендаций для пользователя.

    - **user_id**: ID пользователя
    - **top_k**: количество рекомендаций (1-100, по умолчанию 10)
    - **model_type**: тип модели — `popular` или `tfidf`
    """
    if model_type not in _predictors:
        raise HTTPException(
            status_code=503,
            detail=f"Модель '{model_type}' не загружена. "
                   f"Доступны: {list(_predictors.keys())}",
        )

    predictor = _predictors[model_type]

    try:
        recs = predictor.predict(user_id, top_k=top_k)
    except Exception as e:
        logger.error(f"Ошибка при предсказании для user_id={user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Счётчик предсказаний
    PREDICTION_COUNTER.labels(model_type=model_type).inc()

    return RecommendResponse(
        user_id=user_id,
        recommendations=[RecommendItem(**r) for r in recs],
        model_type=model_type,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)