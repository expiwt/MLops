# -*- coding: utf-8 -*-
"""
DriftReport — генерация HTML-отчётов о дрейфе для веб-интерфейса.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from .detector import DriftReportData

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports" / "drift"


def generate_html(report: DriftReportData) -> str:
    """Генерирует HTML-страницу отчёта о дрейфе."""
    has_drift = report.has_any_drift

    # Собираем все сигналы
    all_signals = report.data_drifts + report.target_drifts
    drift_rows = ""
    for s in all_signals:
        badge = "⚠️ ДРЕЙФ" if s.is_drift else "✓ OK"
        cls = "table-danger" if s.is_drift else "table-success"
        drift_rows += f"""
        <tr class="{cls}">
            <td>{s.feature}</td>
            <td>{s.drift_type}</td>
            <td>{s.statistic:.4f}</td>
            <td>{s.p_value if s.p_value is not None else '—'}</td>
            <td>{s.threshold}</td>
            <td>{badge}</td>
            <td><small>{s.description}</small></td>
        </tr>"""

    # Метрики
    metrics_train_str = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in report.metrics_train.items()
    )
    metrics_test_str = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in report.metrics_test.items()
    )

    # Деградация
    deg_str = "".join(
        f"<tr><td>{k}</td><td>{v:+.2%}</td></tr>" for k, v in report.metric_degradation.items()
    )

    concept_badge = "⚠️ ДРЕЙФ" if report.concept_drift else "✓ OK"
    concept_cls = "alert-danger" if report.concept_drift else "alert-success"

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Drift Report — Kion RecSys</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
    <div class="container py-4">
        <h1 class="mb-4">📊 Drift Detection Report</h1>
        <p class="text-muted">Generated: {report.timestamp}</p>

        <!-- Overview -->
        <div class="alert {concept_cls}">
            <h4 class="alert-heading">{'⚠️ Обнаружен дрейф' if has_drift else '✅ Дрейфа нет'}</h4>
            <p>Train period: <strong>{report.train_period[0]} → {report.train_period[1]}</strong></p>
            <p>Test period: <strong>{report.test_period[0]} → {report.test_period[1]}</strong></p>
            <p>Data drift signals: <strong>{len([s for s in all_signals if s.is_drift])}/{len(all_signals)}</strong></p>
            <p>Concept drift: <strong>{concept_badge}</strong></p>
        </div>

        <!-- Drift signals table -->
        <div class="card mb-4">
            <div class="card-header"><h5 class="mb-0">🔬 Drift Signals</h5></div>
            <div class="card-body p-0">
                <table class="table table-striped mb-0">
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Type</th>
                            <th>Statistic</th>
                            <th>p-value</th>
                            <th>Threshold</th>
                            <th>Status</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>{drift_rows}</tbody>
                </table>
            </div>
        </div>

        <!-- Model metrics comparison -->
        <div class="row">
            <div class="col-md-6 mb-4">
                <div class="card">
                    <div class="card-header"><h5 class="mb-0">📈 Metrics — Train</h5></div>
                    <div class="card-body p-0">
                        <table class="table mb-0">
                            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                            <tbody>{metrics_train_str}</tbody>
                        </table>
                    </div>
                </div>
            </div>
            <div class="col-md-6 mb-4">
                <div class="card">
                    <div class="card-header"><h5 class="mb-0">📉 Metrics — Test</h5></div>
                    <div class="card-body p-0">
                        <table class="table mb-0">
                            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                            <tbody>{metrics_test_str}</tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- Degradation -->
        <div class="card mb-4">
            <div class="card-header"><h5 class="mb-0">📉 Metric Degradation</h5></div>
            <div class="card-body p-0">
                <table class="table mb-0">
                    <thead><tr><th>Metric</th><th>Change</th></tr></thead>
                    <tbody>{deg_str}</tbody>
                </table>
            </div>
        </div>

        <p class="text-muted">
            <small>Kion RecSys — MLOps Drift Detection Report</small>
        </p>
    </div>
</body>
</html>"""
    return html


def save_report(report: DriftReportData) -> Path:
    """Сохраняет HTML + JSON отчёт в reports/drift/."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    html_path = REPORTS_DIR / f"drift_report_{ts}.html"
    html_path.write_text(generate_html(report), encoding="utf-8")
    logger.info(f"HTML отчёт сохранён: {html_path}")

    json_path = REPORTS_DIR / f"drift_report_{ts}.json"
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"JSON отчёт сохранён: {json_path}")

    # Latest symlink
    latest_html = REPORTS_DIR / "latest.html"
    if latest_html.exists():
        latest_html.unlink()
    latest_html.symlink_to(html_path.name)

    latest_json = REPORTS_DIR / "latest.json"
    if latest_json.exists():
        latest_json.unlink()
    latest_json.symlink_to(json_path.name)

    return html_path
