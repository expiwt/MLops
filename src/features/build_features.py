import click
import logging
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
from pathlib import Path

def get_coo_matrix(df, user_col, item_col, users_mapping, items_mapping):
    """Создает разреженную матрицу взаимодействий."""
    weights = np.ones(len(df), dtype=np.float32)
    interaction_matrix = sp.coo_matrix((
        weights, 
        (
            df[user_col].map(users_mapping.get), 
            df[item_col].map(items_mapping.get)
        )
    ))
    return interaction_matrix

@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def main(input_dir, output_dir):
    logger = logging.getLogger(__name__)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Загрузка обработанных данных...")
    interactions = pd.read_csv(input_dir / 'interactions_processed.csv')
    
    # 1. Создаем маппинги (ID -> Index)
    # Это важно сохранить, чтобы потом использовать в сервисе на FastAPI
    logger.info("Создание маппингов пользователей и айтемов...")
    users_inv_mapping = dict(enumerate(interactions['user_id'].unique()))
    users_mapping = {v: k for k, v in users_inv_mapping.items()}

    items_inv_mapping = dict(enumerate(interactions['item_id'].unique()))
    items_mapping = {v: k for k, v in items_inv_mapping.items()}

    # 2. Генерируем матрицу для TF-IDF (implicit)
    logger.info("Генерация разреженной матрицы (COO)...")
    full_train_matrix = get_coo_matrix(
        interactions, 
        'user_id', 'item_id', 
        users_mapping, items_mapping
    )

    # 3. Сохраняем результаты
    logger.info(f"Сохранение признаков в {output_dir}")
    
    # Сохраняем матрицу (в формате .npz)
    sp.save_npz(output_dir / 'train_matrix.npz', full_train_matrix.tocsr())
    
    # Сохраняем маппинги (в формате .pkl), они понадобятся для инференса
    with open(output_dir / 'users_mapping.pkl', 'wb') as f:
        pickle.dump(users_mapping, f)
    with open(output_dir / 'items_inv_mapping.pkl', 'wb') as f:
        pickle.dump(items_inv_mapping, f)

    logger.info("Этап build_features завершен.")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()