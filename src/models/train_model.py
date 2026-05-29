# обучение моделей + MLflow

import sys
from pathlib import Path

# Корень проекта для импорта src (при прямом запуске через CliRunner)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import click
import logging
import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
import pickle

from implicit.nearest_neighbours import TFIDFRecommender
from src.models.baseline import PopularRecommender
from src.models.metrics import compute_metrics

# MLflow может не загрузиться без pkg_resources (setuptools)
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("WARNING: MLflow не загружен (продолжаем без трекинга)")

    class FakeSklearn:
        """Заглушка для mlflow.sklearn."""
        @staticmethod
        def log_model(model, name, registered_model_name=None):
            pass

    class FakeMLflow:
        """Заглушка для mlflow."""
        @staticmethod
        def log_metrics(metrics):
            pass

        @staticmethod
        def log_param(key, value):
            pass

        @staticmethod
        def set_experiment(name):
            pass

        @staticmethod
        def start_run(run_name=None):
            return type('ContextManager', (), {
                '__enter__': lambda s: s,
                '__exit__': lambda s, *a: None,
            })()

        class sklearn:
            log_model = staticmethod(FakeSklearn.log_model)

    mlflow = FakeMLflow()

logger = logging.getLogger(__name__)


@click.command()
@click.argument('processed_data_path', type=click.Path(exists=True))
@click.argument('features_path', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path())
@click.option('--model_type', default='popular', help='Model type: popular or tfidf')
def train(processed_data_path, features_path, model_path, model_type):
    """Обучение модели рекомендаций."""
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("RecSys_Experiments")

    with mlflow.start_run(run_name=f"Run_{model_type}"):
        # 1. Загрузка данных для валидации
        df = pd.read_csv(processed_data_path, parse_dates=['last_watch_dt'])
        last_date = df['last_watch_dt'].max()
        train_df = df[df['last_watch_dt'] < last_date]
        test_df = df[df['last_watch_dt'] == last_date]
        users_test = test_df['user_id'].unique()

        # 2. Обучение выбранной модели
        if model_type == 'tfidf':
            logger.info("Загрузка матриц для TF-IDF...")
            train_matrix = sp.load_npz(Path(features_path) / 'train_matrix.npz')
            with open(Path(features_path) / 'users_mapping.pkl', 'rb') as f:
                users_mapping = pickle.load(f)
            with open(Path(features_path) / 'items_inv_mapping.pkl', 'rb') as f:
                items_inv_mapping = pickle.load(f)

            logger.info("Обучение TF-IDF Recommender...")
            model = TFIDFRecommender(K=10)
            model.fit(train_matrix.tocsr())

            test_user_indices = [users_mapping[u] for u in users_test if u in users_mapping]
            ids, scores = model.recommend(test_user_indices, train_matrix.tocsr()[test_user_indices], N=10)

            recs_list = [[items_inv_mapping[idx] for idx in user_recs] for user_recs in ids]
            final_users = [u for u in users_test if u in users_mapping]

        else:
            logger.info("Обучение Popular Recommender...")
            model = PopularRecommender(days=7, dt_column='last_watch_dt')
            model.fit(train_df)
            recs_list = model.recommend(users_test, N=10)
            final_users = users_test

        # 3. Подготовка recs_df для метрик
        recs_df = pd.DataFrame({'user_id': final_users, 'item_id': recs_list})
        recs_df = recs_df.explode('item_id')
        recs_df['rank'] = recs_df.groupby('user_id').cumcount() + 1

        # 4. Расчет метрик
        metrics = compute_metrics(train_df, test_df, recs_df, top_N=10)
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_algo", model_type)

        # 5. Сохранение
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        if MLFLOW_AVAILABLE:
            mlflow.sklearn.log_model(model, "model", registered_model_name=f"{model_type}_model")

        logger.info(f"Завершено. {model_type} Metrics: {metrics}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    train()
