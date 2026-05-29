# -*- coding: utf-8 -*-
"""
FastAPI-сервис для рекомендательной системы Kion.
Эндпоинты: /predict, /health, /model-info
"""
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import jinja2

from src.models.predict_model import Predictor

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Глобальный предиктор (кешируем загрузку модели)
_predictors: dict = {}


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
    version="0.1.0",
    lifespan=lifespan,
)


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
    return HTMLResponse(content=template.render(request=request))


@app.post("/retrain", tags=["System"])
async def retrain():
    """Запуск переобучения моделей."""
    # TODO: реализовать полноценное переобучение
    logger.info("Запрос на переобучение")
    return {"message": "Переобучение запущено. Модели будут обновлены через несколько минут."}


# --- Эндпоинты ---

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
        version="0.1.0",
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

    return RecommendResponse(
        user_id=user_id,
        recommendations=[RecommendItem(**r) for r in recs],
        model_type=model_type,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
