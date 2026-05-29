"""Smoke-тесты для predict_model.py"""
import pytest
from pathlib import Path
from src.models.predict_model import Predictor


def test_predictor_popular_loads():
    """Проверяем, что PopularRecommender загружается и отдаёт рекомендации."""
    pred = Predictor(model_type='popular')
    recs = pred.predict(user_id=123, top_k=5)

    assert len(recs) == 5
    assert all('item_id' in r for r in recs)
    assert all('rank' in r for r in recs)
    assert recs[0]['rank'] == 1


@pytest.mark.slow
def test_predictor_tfidf_loads():
    """TF-IDF модель (может быть медленной из-за загрузки фичей)."""
    pred = Predictor(model_type='tfidf')
    recs = pred.predict(user_id=123, top_k=3)

    assert len(recs) <= 3
    assert all('item_id' in r for r in recs)
