# -*- coding: utf-8 -*-
#инференс
import logging
import pickle
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Пути по умолчанию (относительно корня проекта)
MODELS_DIR = Path(__file__).resolve().parents[2] / 'models'
FEATURES_DIR = Path(__file__).resolve().parents[2] / 'data' / 'features'


class Predictor:
    """
    Класс для инференса моделей рекомендательной системы Kion.
    Поддерживает PopularRecommender и TFIDFRecommender.
    """

    def __init__(self, model_type: str = 'popular'):
        self.model_type = model_type
        self.model = None
        self.features = None
        self._load_model()
        self._load_features()

    def _load_model(self):
        """Загружает модель из папки models/."""
        model_paths = {
            'popular': MODELS_DIR / 'popular.pkl',
            'tfidf': MODELS_DIR / 'tfidf.pkl',
        }

        path = model_paths.get(self.model_type)
        if not path or not path.exists():
            raise FileNotFoundError(f"Модель {self.model_type} не найдена: {path}")

        logger.info(f"Загрузка модели {self.model_type} из {path}")
        self.model = joblib.load(path)

    def _load_features(self):
        """Загружает фичи, необходимые для инференса."""
        if self.model_type == 'tfidf':
            logger.info("Загрузка фичей для TF-IDF...")
            self.features = {}
            self.features['train_matrix'] = sp.load_npz(FEATURES_DIR / 'train_matrix.npz')
            with open(FEATURES_DIR / 'users_mapping.pkl', 'rb') as f:
                self.features['users_mapping'] = pickle.load(f)
            with open(FEATURES_DIR / 'items_inv_mapping.pkl', 'rb') as f:
                self.features['items_inv_mapping'] = pickle.load(f)

    def predict(self, user_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Возвращает top-K рекомендаций для пользователя.

        Args:
            user_id: ID пользователя.
            top_k: Количество рекомендаций.

        Returns:
            Список словарей: [{'item_id': ..., 'rank': ...}, ...]
        """
        if self.model_type == 'popular':
            return self._predict_popular(top_k)
        elif self.model_type == 'tfidf':
            return self._predict_tfidf(user_id, top_k)
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")

    def _predict_popular(self, top_k: int) -> List[Dict[str, Any]]:
        """Рекомендации популярного — одинаковые для всех."""
        recs = self.model.recommend(None, N=top_k)
        return [
            {'item_id': int(item), 'rank': rank + 1}
            for rank, item in enumerate(recs)
        ]

    def _predict_tfidf(self, user_id: int, top_k: int) -> List[Dict[str, Any]]:
        """Рекомендации TF-IDF — персональные."""
        users_mapping = self.features['users_mapping']
        items_inv_mapping = self.features['items_inv_mapping']
        train_matrix = self.features['train_matrix']

        if user_id not in users_mapping:
            logger.warning(f"Пользователь {user_id} не найден в обучающей выборке")
            return []

        user_idx = users_mapping[user_id]
        user_vector = train_matrix[user_idx]
        ids, scores = self.model.recommend(
            [user_idx], user_vector, N=top_k
        )

        return [
            {'item_id': int(items_inv_mapping[idx]), 'rank': rank + 1, 'score': float(scores[0][rank])}
            for rank, idx in enumerate(ids[0])
        ]


def predict(user_ids: List[int], model_type: str = 'popular', top_k: int = 10) -> pd.DataFrame:
    """
    Удобная функция для batch-инференса.

    Args:
        user_ids: Список ID пользователей.
        model_type: 'popular' или 'tfidf'.
        top_k: Количество рекомендаций на пользователя.

    Returns:
        DataFrame с колонками: user_id, item_id, rank.
    """
    predictor = Predictor(model_type=model_type)
    all_recs = []

    for uid in user_ids:
        recs = predictor.predict(uid, top_k=top_k)
        for r in recs:
            all_recs.append({
                'user_id': uid,
                'item_id': r['item_id'],
                'rank': r['rank'],
            })

    return pd.DataFrame(all_recs)
