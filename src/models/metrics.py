import numpy as np
import pandas as pd

def calculate_novelty(train_interactions, recommendations, top_n): 
    users = recommendations['user_id'].unique()
    n_users = train_interactions['user_id'].nunique()
    n_users_per_item = train_interactions.groupby('item_id')['user_id'].nunique()

    recommendations = recommendations.loc[recommendations['rank'] <= top_n].copy()
    recommendations['n_users_per_item'] = recommendations['item_id'].map(n_users_per_item).fillna(1)
    recommendations['item_novelty'] = -np.log2(recommendations['n_users_per_item'] / n_users)

    miuf_at_k = recommendations.groupby('user_id')['item_novelty'].mean()
    return miuf_at_k.reindex(users).mean()

def compute_metrics(train, test, recs, top_N):
    # Объединяем прогноз с фактом
    test_recs = test.set_index(['user_id', 'item_id']).join(recs.set_index(['user_id', 'item_id']))
    test_recs = test_recs.sort_values(by=['user_id', 'rank'])

    # Считаем количество реальных просмотров у каждого юзера в тесте
    test_recs['users_item_count'] = test_recs.groupby(level='user_id')['rank'].transform(np.size)
    
    # Флаг попадания (был ли рекомендованный айтем реально просмотрен)
    test_recs['hit'] = test_recs['rank'].notnull()
    users_count = test_recs.index.get_level_values('user_id').nunique()
    
    # --- Расчет MAP ---
    test_recs['cumulative_rank'] = test_recs.groupby(level='user_id').cumcount() + 1
    test_recs['cumulative_rank'] = test_recs['cumulative_rank'] / test_recs['rank']
    map_f = (test_recs['cumulative_rank'] / test_recs['users_item_count']).sum() / users_count

    # --- Расчет Precision & Recall ---
    # Precision: сколько из N рекомендаций были полезны
    precision = test_recs['hit'].sum() / (users_count * top_N)
    # Recall: какую долю из того, что юзер посмотрел, мы угадали
    recall = test_recs['hit'].sum() / test_recs['users_item_count'].sum()

    return {
        f'MAP_{top_N}': map_f,
        f'Novelty_{top_N}': calculate_novelty(train, recs, top_N),
        f'Precision_{top_N}': precision,
        f'Recall_{top_N}': recall
    }