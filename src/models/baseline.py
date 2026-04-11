import pandas as pd
import numpy as np
from itertools import islice, cycle

class PopularRecommender():
    def __init__(self, max_K=10, days=30, item_column='item_id', dt_column='date'):
        self.max_K = max_K
        self.days = days
        self.item_column = item_column
        self.dt_column = dt_column
        self.recommendations = []
        
    def fit(self, df):
        # Логика из твоего ноутбука: берем популярное за последние N дней
        # Используем normalize() и DateOffset для точности
        min_date = df[self.dt_column].max().normalize() - pd.DateOffset(days=self.days)
        self.recommendations = df.loc[df[self.dt_column] > min_date, self.item_column] \
                                 .value_counts() \
                                 .head(self.max_K) \
                                 .index.values
    
    def recommend(self, users=None, N=10):
        """
        Метод должен называться именно recommend, 
        чтобы совпадать с интерфейсом библиотек типа implicit.
        """
        recs = self.recommendations[:N]
        if users is None:
            return recs
        else:
            # Возвращаем список рекомендаций для каждого пользователя
            return list(islice(cycle([recs]), len(users)))