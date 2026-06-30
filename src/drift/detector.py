# -*- coding: utf-8 -*-
"""
DriftDetector — обнаружение data drift, concept drift и target drift.
- Data drift: KS-тест для числовых признаков (watched_pct, total_dur),
  JS divergence / Chi2 для категориальных (item_id).
- Target drift: распределение watched_pct.
- Concept drift: сравнение метрик модели на разных временных срезах.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, entropy, ks_2samp

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Thresholds
# ──────────────────────────────────────────────
DRIFT_THRESHOLDS = {
    "ks_pvalue": 0.05,          # KS p-value ниже → дрейф
    "js_threshold": 0.1,        # JS divergence выше → дрейф
    "chi2_pvalue": 0.05,        # Chi2 p-value ниже → дрейф
    "metric_degradation": 0.05, # MAP упал на >5% → concept drift
}


# ──────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────
@dataclass
class DriftSignal:
    feature: str
    drift_type: str          # "data_drift" | "target_drift" | "concept_drift"
    statistic: float
    p_value: Optional[float]
    threshold: float
    is_drift: bool
    ref_mean: float
    cur_mean: float
    description: str = ""


@dataclass
class DriftReportData:
    """Полный отчёт о дрейфе."""
    timestamp: str
    train_period: Tuple[str, str]
    test_period: Tuple[str, str]
    drifts: List[DriftSignal] = field(default_factory=list)
    metrics_train: Dict[str, float] = field(default_factory=dict)
    metrics_test: Dict[str, float] = field(default_factory=dict)
    concept_drift: bool = False
    metric_degradation: Dict[str, float] = field(default_factory=dict)

    @property
    def has_any_drift(self) -> bool:
        return any(s.is_drift for s in self.drifts) or self.concept_drift

    @property
    def drift_count(self) -> int:
        return sum(1 for s in self.drifts if s.is_drift) + (1 if self.concept_drift else 0)

    @property
    def data_drifts(self) -> List[DriftSignal]:
        return [s for s in self.drifts if s.drift_type == "data_drift"]

    @property
    def target_drifts(self) -> List[DriftSignal]:
        return [s for s in self.drifts if s.drift_type == "target_drift"]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "train_period": list(self.train_period),
            "test_period": list(self.test_period),
            "has_any_drift": self.has_any_drift,
            "drift_count": self.drift_count,
            "concept_drift": self.concept_drift,
            "metric_degradation": self.metric_degradation,
            "data_drifts": [
                {"feature": s.feature, "drift_type": s.drift_type,
                 "statistic": s.statistic, "p_value": s.p_value,
                 "threshold": s.threshold, "is_drift": s.is_drift,
                 "ref_mean": s.ref_mean, "cur_mean": s.cur_mean,
                 "description": s.description}
                for s in self.data_drifts
            ],
            "target_drifts": [
                {"feature": s.feature, "drift_type": s.drift_type,
                 "statistic": s.statistic, "p_value": s.p_value,
                 "threshold": s.threshold, "is_drift": s.is_drift,
                 "ref_mean": s.ref_mean, "cur_mean": s.cur_mean,
                 "description": s.description}
                for s in self.target_drifts
            ],
            "metrics_train": self.metrics_train,
            "metrics_test": self.metrics_test,
        }


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two discrete distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2))


def _chi2_drift(ref_series: pd.Series, cur_series: pd.Series,
                feature_name: str = "item_id") -> DriftSignal:
    """Chi-squared test for categorical feature drift (топ-50 + other)."""
    threshold = DRIFT_THRESHOLDS["chi2_pvalue"]
    top = ref_series.value_counts().head(50).index.tolist()
    ref_counts = ref_series.where(ref_series.isin(top), "other").value_counts()
    cur_counts = cur_series.where(cur_series.isin(top), "other").value_counts()
    all_cats = list(set(ref_counts.index) | set(cur_counts.index))
    ref_arr = np.array([ref_counts.get(c, 0) for c in all_cats])
    cur_arr = np.array([cur_counts.get(c, 0) for c in all_cats])

    if ref_arr.sum() == 0 or cur_arr.sum() == 0:
        return DriftSignal(
            feature=feature_name, drift_type="data_drift",
            statistic=0.0, p_value=1.0,
            threshold=threshold, is_drift=False,
            ref_mean=0.0, cur_mean=0.0,
            description="Недостаточно данных для Chi2 теста",
        )
    stat, p_val, _, _ = chi2_contingency(pd.DataFrame({"ref": ref_arr, "cur": cur_arr}))
    return DriftSignal(
        feature=feature_name, drift_type="data_drift",
        statistic=float(stat), p_value=float(p_val),
        threshold=threshold, is_drift=bool(p_val < threshold),
        ref_mean=float(ref_arr.mean()), cur_mean=float(cur_arr.mean()),
        description=f"Chi2 p-value={p_val:.4f} {'⚠️ ДРЕЙФ' if p_val < threshold else '✓ OK'}",
    )


def _ks_drift(ref: pd.Series, cur: pd.Series,
              feature_name: str = "", drift_type: str = "data_drift") -> DriftSignal:
    """KS-тест для числового признака."""
    threshold = DRIFT_THRESHOLDS["ks_pvalue"]
    ref_clean = ref.dropna()
    cur_clean = cur.dropna()
    if len(ref_clean) < 10 or len(cur_clean) < 10:  # ✅ исправлено: всё в одну строку
        return DriftSignal(
            feature=feature_name, drift_type=drift_type,
            statistic=0.0, p_value=1.0,
            threshold=threshold, is_drift=False,
            ref_mean=float(ref_clean.mean()) if len(ref_clean) > 0 else 0.0,
            cur_mean=float(cur_clean.mean()) if len(cur_clean) > 0 else 0.0,
            description="Недостаточно данных для KS теста",
        )
    stat, p_val = ks_2samp(ref_clean, cur_clean)
    return DriftSignal(
        feature=feature_name, drift_type=drift_type,
        statistic=float(stat), p_value=float(p_val),
        threshold=threshold, is_drift=bool(p_val < threshold),
        ref_mean=float(ref_clean.mean()), cur_mean=float(cur_clean.mean()),
        description=f"KS stat={stat:.4f} p={p_val:.4f} "
                    f"{'⚠️ ДРЕЙФ' if p_val < threshold else '✓ OK'}",
    )


def _js_drift(ref_dist: np.ndarray, cur_dist: np.ndarray,
              feature_name: str, drift_type: str = "data_drift") -> DriftSignal:
    """JS divergence для сравнения распределений."""
    threshold = DRIFT_THRESHOLDS["js_threshold"]
    js = _js_divergence(ref_dist, cur_dist)
    return DriftSignal(
        feature=feature_name, drift_type=drift_type,
        statistic=float(js), p_value=None,
        threshold=threshold, is_drift=bool(js > threshold),
        ref_mean=float(ref_dist.mean()), cur_mean=float(cur_dist.mean()),
        description=f"JS divergence={js:.4f} "
                    f"{'⚠️ ДРЕЙФ' if js > threshold else '✓ OK'}",
    )


# ──────────────────────────────────────────────
#  Основной детектор
# ──────────────────────────────────────────────

class DriftDetector:
    """
    DriftDetector: загружает данные за два периода,
    считает дрейфы и метрики модели на каждом периоде.

    Идея:
      - Data drift: сравниваем распределения признаков train vs test периода.
      - Target drift: сравниваем распределение watched_pct.
      - Concept drift: обучаем модель ТОЛЬКО на train_period, 
        затем предсказываем на test_period (без дообучения).
        Если MAP@10 упал >5% относительно baseline — concept drift.
    """

    def __init__(self, data_path: str = "data/processed/interactions_processed.csv"):
        self.data_path = Path(data_path)
        self.df: pd.DataFrame = None
        self._load()

    def _load(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Файл данных не найден: {self.data_path}")
        self.df = pd.read_csv(self.data_path, parse_dates=["last_watch_dt"])
        logger.info(f"Загружено {len(self.df)} интеракций "
                    f"({self.df['last_watch_dt'].min()} → {self.df['last_watch_dt'].max()})")

    # ──────────────────────────────────────
    #  Numerical drift
    # ──────────────────────────────────────
    def check_numerical_drift(self, ref: pd.DataFrame, cur: pd.DataFrame) -> List[DriftSignal]:
        results = []
        for col in ["watched_pct", "total_dur"]:
            if col not in ref.columns or col not in cur.columns:
                continue
            results.append(_ks_drift(ref[col], cur[col], feature_name=col))
        return results

    # ──────────────────────────────────────
    #  Categorical drift
    # ──────────────────────────────────────
    def check_categorical_drift(self, ref: pd.DataFrame, cur: pd.DataFrame) -> List[DriftSignal]:
        results = []
        # JS divergence для популярности айтемов
        ref_top = ref["item_id"].value_counts(normalize=True).head(100)
        cur_top = cur["item_id"].value_counts(normalize=True).head(100)
        all_items = list(set(ref_top.index) | set(cur_top.index))
        ref_dist = np.array([ref_top.get(i, 0) for i in all_items])
        cur_dist = np.array([cur_top.get(i, 0) for i in all_items])
        results.append(_js_drift(ref_dist, cur_dist, feature_name="item_id_popularity"))

        # Chi2 для топ-50 айтемов
        results.append(_chi2_drift(ref["item_id"], cur["item_id"]))

        return results

    # ──────────────────────────────────────
    #  Target drift
    # ──────────────────────────────────────
    def check_target_drift(self, ref: pd.DataFrame, cur: pd.DataFrame) -> List[DriftSignal]:
        return [_ks_drift(ref["watched_pct"], cur["watched_pct"],
                          feature_name="watched_pct (target)",
                          drift_type="target_drift")]

    # ──────────────────────────────────────
    #  Concept drift (исправленная логика)
    # ──────────────────────────────────────
    def _train_predict_and_evaluate(self, train_df: pd.DataFrame, eval_df: pd.DataFrame,
                                    model_type: str = "popular",
                                    top_N: int = 10) -> Dict[str, float]:
        """
        Обучить модель на train_df, предсказать на eval_df, вернуть метрики.
        eval_df НЕ участвует в обучении — это pure production scenario.
        """
        from src.models.baseline import PopularRecommender
        from src.models.metrics import compute_metrics

        # Обучаем на всём train_df
        model = PopularRecommender(days=7, dt_column='last_watch_dt')
        model.fit(train_df)

        # Тестовые пользователи — те, кто есть в eval_df
        users_eval = eval_df['user_id'].unique()

        # Предсказания этой моделью (без дообучения!)
        recs_list = model.recommend(users_eval, N=top_N)

        recs_df = pd.DataFrame({
            'user_id': users_eval,
            'item_id': recs_list
        })
        recs_df = recs_df.explode('item_id')
        recs_df['rank'] = recs_df.groupby('user_id').cumcount() + 1

        # Метрики считаем на eval_df как ground truth
        metrics = compute_metrics(train_df, eval_df, recs_df, top_N=top_N)
        return metrics

    def check_concept_drift(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[bool, Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Concept drift: обучаем на train_period → предсказываем на test_period.
        Сравниваем MAP@10 на train_period (baseline) vs test_period (production).

        Возвращает: (is_drift, metrics_baseline, metrics_production, degradation)
        """
        logger.info("Расчёт baseline метрик (train → train test split)...")
        # Baseline: модель на train_period, оценка на train_period
        # Берём последний день train_period как валидацию
        last_date_train = train_df['last_watch_dt'].max()
        train_train = train_df[train_df['last_watch_dt'] < last_date_train]
        train_test = train_df[train_df['last_watch_dt'] == last_date_train]
        metrics_baseline = self._train_predict_and_evaluate(train_train, train_test)

        logger.info("Расчёт production метрик (train → test period)...")
        # Production: та же модель (обучена на train_period) → предсказание на test_period
        # Обучаем на train_period целиком, предсказываем на test_period
        metrics_production = self._train_predict_and_evaluate(train_df, test_df)

        # Сравнение
        degradation = {}
        for key in metrics_baseline:
            if key in metrics_production and metrics_baseline[key] != 0:
                delta = (metrics_production[key] - metrics_baseline[key]) / abs(metrics_baseline[key])
                degradation[key] = round(float(delta), 4)

        map_key = "MAP_10"
        is_drift = False
        if map_key in degradation:
            is_drift = degradation[map_key] < -DRIFT_THRESHOLDS["metric_degradation"]

        logger.info(f"Concept drift: baseline={metrics_baseline}, "
                    f"production={metrics_production}, "
                    f"degradation={degradation}, drift={'⚠️' if is_drift else '✓'}")
        return is_drift, metrics_baseline, metrics_production, degradation

    # ──────────────────────────────────────
    #  Full check
    # ──────────────────────────────────────
    def run_full_check(self, train_start: str, train_end: str,
                       test_start: str, test_end: str) -> DriftReportData:
        logger.info(f"Drift check: train=[{train_start}, {train_end}), "
                    f"test=[{test_start}, {test_end})")

        train_df = self.df.query("last_watch_dt >= @train_start and last_watch_dt < @train_end").copy()
        test_df = self.df.query("last_watch_dt >= @test_start and last_watch_dt < @test_end").copy()

        logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

        if len(train_df) < 100 or len(test_df) < 100:
            raise ValueError(f"Слишком мало данных: train={len(train_df)}, test={len(test_df)}")

        # Data drift
        data_drifts = self.check_numerical_drift(train_df, test_df)
        data_drifts += self.check_categorical_drift(train_df, test_df)
        logger.info(f"Data drifts: {len([d for d in data_drifts if d.is_drift])}/{len(data_drifts)}")

        # Target drift
        target_drifts = self.check_target_drift(train_df, test_df)
        logger.info(f"Target drifts: {len([d for d in target_drifts if d.is_drift])}/{len(target_drifts)}")

        # Concept drift — обучаем на train_period, предсказываем на test_period
        logger.info("Вычисление concept drift...")
        concept_drift, metrics_baseline, metrics_production, degradation = self.check_concept_drift(
            train_df, test_df
        )  # ✅ исправлено: перенос на предыдущую строку

        report = DriftReportData(
            timestamp=datetime.now().isoformat(),
            train_period=(train_start, train_end),
            test_period=(test_start, test_end),
            drifts=data_drifts + target_drifts,
            metrics_train=metrics_baseline,
            metrics_test=metrics_production,
            concept_drift=concept_drift,
            metric_degradation=degradation,
        )
        return report