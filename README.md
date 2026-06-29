# Kion RecSys — MLOps Project

Рекомендательная система для онлайн-кинотеатра Kion с полным MLOps-циклом:
от версионирования данных до мониторинга в Kubernetes.

## Содержание

- [Архитектура](#архитектура)
- [Состав проекта](#состав-проекта)
- [Быстрый старт](#быстрый-старт)
- [Мониторинг (Prometheus + Grafana)](#мониторинг-prometheus--grafana)
- [Модели и MLflow](#модели-и-mlflow)
- [Web UI](#web-ui)
- [API Endpoints](#api-endpoints)
- [CI/CD](#cicd)

---

## Архитектура

```
┌──────────────────────────────────────────────────────────┐
│                    GitHub Actions                         │
│       Lint → Test → Build Docker → Push to Registry      │
└───────────────────────┬──────────────────────────────────┘
                        │ CD (ArgoCD sync)
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Kubernetes (kind)                               │
│                                                                   │
│  ┌─────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │  FastAPI     │  │  MLflow  │  │Portainer │  │  Prometheus  │ │
│  │  RecSys API  │  │ Tracking │  │ K8s UI   │  │  + Grafana   │ │
│  │  :8000      │  │ :5000   │  │ :9000   │  │  :9090/:3000 │ │
│  └──────┬───────┘  └──────────┘  └──────────┘  └──────────────┘ │
│         │                     ns:recsys         ns:monitoring    │
└──────────────────────────────────────────────────────────────────┘

                 ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
                  Локально (docker-compose)
                 │  + Prometheus :9090             │
                  + Grafana    :3000
                 └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
```

**Компоненты:**

| Компонент | Назначение |
|-----------|-----------|
| **FastAPI сервис** | Выдаёт рекомендации, Web UI, Prometheus метрики |
| **MLflow** | Трекинг экспериментов, регистрация моделей |
| **Prometheus + Grafana** | Сбор метрик (RPS, latency, статус-коды, предсказания, метрики моделей) и визуализация |
| **kube-prometheus-stack** | Оператор для Prometheus, автоматическое обнаружение ServiceMonitor |
| **Portainer** | Веб-интерфейс управления Kubernetes |
| **Kind** | Локальный Kubernetes-кластер в Docker |
| **DVC** | Версионирование данных и моделей |

---

## Состав проекта

```
├── app/                        # ★ FastAPI сервис
│   ├── main.py                 #  Эндпоинты, Prometheus метрики
│   ├── templates/index.html    #  Web UI (инференс, история, retrain)
│   └── static/                 #  Статика для UI
│
├── src/                        # ★ ML код
│   ├── data/make_dataset.py    #  Загрузка данных
│   ├── features/               #  Сборка sparse-матриц для TF-IDF
│   ├── models/
│   │   ├── train_model.py      #  Обучение (click + MLflow трекинг)
│   │   ├── predict_model.py    #  Инференс (Predictor класс)
│   │   ├── baseline.py         #  PopularRecommender
│   │   └── metrics.py          #  MAP, Precision, Recall, Novelty
│
├── k8s/                        # ★ Kubernetes манифесты
│   ├── namespace.yaml          #  namespace: recsys
│   ├── deployment.yaml         #  recsys-api + mlflow (deployments + services)
│   ├── configmap.yaml          #  Конфигурация + secrets
│   ├── ingress.yaml            #  Внешний доступ
│   ├── hpa.yaml                #  Автоскалирование
│   ├── pvc.yaml                #  PersistentVolumeClaims
│   ├── network-policy.yaml     #  Сетевая изоляция (доступ для monitoring)
│   ├── resource-quota.yaml     #  Квоты
│   ├── argocd/                 #  ArgoCD Application manifest
│   ├── prometheus/             #  ServiceMonitor для Prometheus Operator
│   └── portainer/              #  Portainer (k8s web UI)
│
├── infra/                      # ★ Дашборды и конфиги мониторинга
│   ├── prometheus/prometheus.yml
│   └── grafana/
│       ├── provisioning/       #  Автоконфиг датасорсов и дашбордов
│       └── dashboards/         #  JSON-дашборд RecSys API
│
├── src/drift/                  # ★ Детекция дрифта
│   ├── detector.py             #  Data / Target / Concept drift
│   └── report.py               #  HTML / JSON отчёты
│
├── tests/                      # ★ Тесты
│   └── test_predict.py         #  Тесты инференса
│
├── data/                       # Данные под DVC
├── models/                     # Обученные модели под DVC
├── reports/drift/              #  Отчёты о дрейфе (HTML + JSON)
├── .github/workflows/ci.yml   # CI/CD
│
├── Dockerfile                  # Сборка образа
├── docker-compose.yml          # Локальный запуск (api + mlflow + prometheus + grafana)
├── requirements.txt            # Зависимости
├── Makefile                    # Команды (k8s, portainer)
└── README.md                   # Эта документация
```

---

## Быстрый старт

### Локально (docker-compose)

```bash
cd ~/MLops
source venv/bin/activate
pip install "dvc[s3]"

# Настройка DVC и загрузка данных
dvc remote modify --local s3-storage access_key_id "ваш_access_key"
dvc remote modify --local s3-storage secret_access_key "ваш_secret_key"
dvc pull

docker-compose up --build
```

| Сервис | Адрес |
|--------|-------|
| Web UI / API | http://localhost:8000 |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/пароль из docker-compose) |

docker-compose down

### Kubernetes (kind)

```bash
# 1. Создать кластер
kind create cluster --name recsys

# 2. Собрать и загрузить образы
docker build -t mlops-recsys-api:latest .
kind load docker-image mlops-recsys-api:latest --name recsys

docker pull ghcr.io/mlflow/mlflow:v3.12.0
docker save ghcr.io/mlflow/mlflow:v3.12.0 | docker exec -i recsys-control-plane ctr -n k8s.io images import -

# 3. Развернуть
make k8s-apply
kubectl apply -f k8s/deployment.yaml
kubectl get pods -n recsys -w

# 4. Порт-форварды
kubectl port-forward -n recsys svc/recsys-api 8000:8000
kubectl port-forward -n recsys svc/mlflow 5000:5000

# 5. Мониторинг
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace
kubectl apply -f k8s/prometheus/

kubectl port-forward -n monitoring svc/prometheus-stack-grafana 3000:80
kubectl port-forward -n monitoring svc/prometheus-stack-kube-prom-prometheus 9090:9090

# Тест API
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict/123?top_k=5&model_type=popular"
```

### Что развёрнуто

| Компонент | Образ | Ресурсы |
|-----------|-------|---------|
| recsys-api | mlops-recsys-api:latest | 512Mi / 2Gi |
| mlflow | ghcr.io/mlflow/mlflow:v3.12.0 | 512Mi / 2Gi |
| portainer | portainer/portainer-ce:latest | 128Mi / 512Mi |
| prometheus | prometheus-stack-kube-prom:3.12.0 | — |
| grafana | grafana:11.1.0 | — |

### ArgoCD

```bash
kubectl apply -f k8s/argocd/application.yaml
```

Также возможно развернуть на **Minikube** или любом другом кластере —
манифесты универсальны.

---

## Мониторинг (Prometheus + Grafana)

Мониторинг разворачивается через **kube-prometheus-stack** (helm-чарт), который включает Prometheus Operator, Prometheus, AlertManager и Grafana.

**Сбор метрик:**

Сервис отдаёт метрики на `/metrics`. ServiceMonitor в `k8s/prometheus/` автоматически настраивает сбор через Prometheus Operator.

| Метрика | Тип | Описание |
|---------|-----|----------|
| `recsys_requests_total` | Counter | Количество запросов (method, endpoint, status) |
| `recsys_request_latency_seconds` | Histogram | Задержка ответов в секундах |
| `recsys_predictions_total` | Counter | Количество выданных рекомендаций (model_type) |
| `recsys_model_metric` | Gauge | Метрики модели: MAP, Precision, Recall, Novelty (model_type, metric) |

**Дашборд Grafana:**

Автоматически импортируется дашборд **RecSys API** (`infra/grafana/dashboards/recsys-dashboard.json`):

| Панель | Показывает |
|--------|-----------|
| RPS | Запросов в секунду |
| Latency | p50 / p95 / p99 времени ответа |
| HTTP Status Codes | 2xx / 4xx / 5xx |
| Predictions by Model | Предсказания по моделям (popular vs tfidf) |
| Model Metrics — Popular | MAP@10, Precision@10, Recall@10, Novelty@10 |
| Model Metrics — TF-IDF | MAP@10, Precision@10, Recall@10, Novelty@10 |

**Установка мониторинга:**

```bash
# Установка kube-prometheus-stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace

# Применить ServiceMonitor для recsys-api
kubectl apply -f k8s/prometheus/

# Доступ к Grafana
kubectl port-forward -n monitoring svc/prometheus-stack-grafana 3000:80
```

Логин: `admin`, пароль:
```bash
kubectl --namespace monitoring get secrets prometheus-stack-grafana \
  -o jsonpath="{.data.admin-password}" | base64 -d ; echo
```

Дашборд импортируется в Grafana через API после установки. JSON в `infra/grafana/dashboards/`.

---

## Drift Detection

Обнаружение **data drift**, **target drift** и **concept drift** для рекомендательной модели.

### Типы дрифта

| Тип | Метод | Что проверяет |
|-----|-------|---------------|
| **Data drift** | KS-тест (числовые), Chi² / JS divergence (категориальные) | Изменилось ли распределение признаков (`watched_pct`, `total_dur`, популярность `item_id`) между временными периодами |
| **Target drift** | KS-тест | Изменилось ли распределение целевой переменной (`watched_pct`) |
| **Concept drift** | Cross-period evaluation | Обучаем модель **только на train_period**, предсказываем **на test_period** (без дообучения). Если MAP@10 упал >5% — concept drift |

### Как это работает

Исходные данные — 5.5 месяцев просмотров (март–август 2021, ~5.4M интеракций).
Drift check делит данные на два временных окна и сравнивает.

**Обычное разбиение (time-based):**
```
Train: все данные, кроме последнего дня
Test:  последний день (22 августа)
→ модель знает распределение, метрики высокие
```

**Drift-разбиение (cross-period):**
```
Train: март–май (3 месяца)
Test:  июль–август (2 месяца), модель НЕ видела эти данные
→ если распределения изменились — метрики падают
```

### Пример результатов

```
                     Normal       Drift        Δ%
MAP_10             0.0900      0.0742     −17.6%  📉
Precision_10       0.0290      0.0569     +96.0%  📈
Recall_10          0.0622      0.0075     −87.9%  📉
```

Модель, обученная на март–май, теряет **17.6% MAP** и **87.9% Recall** при предсказании на июль–август:
- Популярные айтемы сменились (пересечение топ-10: **3/10**)
- Изменилось поведение пользователей (`watched_pct` KS p≈0)

### Пороги срабатывания

| Сигнал | Порог |
|--------|-------|
| KS p-value | < 0.05 |
| JS divergence | > 0.1 |
| Chi² p-value | < 0.05 |
| Concept drift (MAP падение) | > 5% |

### Эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/drift/run` | Запустить drift check (`?train_start=&train_end=&test_start=&test_end=`) |
| `GET` | `/drift/status` | Статус последнего чека |
| `GET` | `/drift/data` | JSON с результатами |
| `GET` | `/drift/report` | HTML-отчёт |
| `GET` | `/drift/history` | История запусков |

### Web UI

На главной странице (http://localhost:8000):
- **Drift Detection Panel** — статус, кнопка запуска, ссылка на отчёт
- **Флаги аномалий** — для каждого признака: ⚠️ ДРЕЙФ / ✅ OK
- **Alert Banner** — красный баннер при обнаружении дрейфа
- **История экспериментов** — хронология запусков
- **Отчёт** — отдельная страница `/drift/report` с полной таблицей сигналов, метрик и деградации

### Пример запуска

```bash
# Запустить drift check (train: март-май, test: июль-август)
curl -X POST 'http://localhost:8000/drift/run?\
  train_start=2021-03-13&train_end=2021-06-01&\
  test_start=2021-07-01&test_end=2021-08-23'

# Статус
curl http://localhost:8000/drift/status

# Открыть HTML-отчёт в браузере
open http://localhost:8000/drift/report
```

### Структура модуля

```
src/drift/
├── __init__.py       # Пакет
├── detector.py       # DriftDetector — все типы дрифта
└── report.py         # Генерация HTML + JSON отчётов
```

---

## Модели и MLflow

### Модели

| Модель | Тип | Описание |
|--------|-----|----------|
| **PopularRecommender** | Content-based | Топ популярных фильмов за последние N дней |
| **TFIDFRecommender** | Collaborative | TF-IDF на user-item матрице, персональные рекомендации |

### Метрики (рассчитываются при обучении)

- **MAP@k** — Mean Average Precision
- **Precision@k** — точность рекомендаций
- **Recall@k** — полнота
- **Novelty** — новизна (насколько рекомендации неочевидны)

### MLflow

Эксперименты логируются в MLflow при каждом обучении:

```bash
kubectl port-forward -n recsys svc/mlflow 5000:5000
# → http://localhost:5000
```

В MLflow сохраняется:
- Метрики: MAP_10, Precision_10, Recall_10, Novelty_10
- Параметры: model_algo (popular / tfidf)
- Модель: registered_model_name (popular_model / tfidf_model)

Модели для инференса загружаются напрямую из `models/*.pkl`.

### Переобучение

```bash
# Через Web UI — кнопка "Запустить переобучение"
# Через API:
curl -X POST http://localhost:8000/retrain
curl http://localhost:8000/retrain/status
```

Результаты переобучения автоматически экспортируются в Prometheus (метрика `recsys_model_metric`) и отображаются в Grafana.

---

## Web UI

Доступен по http://localhost:8000:

- **Форма инференса** — ввод user_id, количества рекомендаций, выбор модели
- **Статус системы** — загруженные модели
- **Кнопка переобучения** — запуск retrain
- **Drift Detection Panel** — статус проверки дрифта, кнопка запуска, ссылка на отчёт
- **Флаги аномалий** — таблица с сигналами data/target/concept drift для каждого признака
- **История экспериментов** — хронология запусков дрифт-чека
- **Alert Banner** — красный/зелёный баннер при обнаружении/отсутствии дрейфа
- **Ссылки** — MLflow, Prometheus, Grafana
- **Автообновление** — каждые 30 секунд

---

## API Endpoints

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/` | Web UI |
| `POST` | `/predict/{user_id}` | Рекомендации (`?top_k=5&model_type=popular`) |
| `GET` | `/health` | Статус сервиса |
| `GET` | `/model-info` | Информация о моделях |
| `GET` | `/metrics` | Prometheus метрики |
| `POST` | `/retrain` | Запуск переобучения |
| `GET` | `/retrain/status` | Статус переобучения |
| `POST` | `/drift/run` | Запуск drift check (`?train_start=&train_end=&test_start=&test_end=`) |
| `GET` | `/drift/status` | Статус последнего drift check |
| `GET` | `/drift/data` | JSON с результатами последнего чека |
| `GET` | `/drift/report` | HTML-отчёт последнего cheka |
| `GET` | `/drift/history` | История всех запусков drift check |

### Пример запроса

```bash
curl -X POST "http://localhost:8000/predict/123?top_k=5&model_type=popular"

# Ответ:
{
  "user_id": 123,
  "recommendations": [
    {"item_id": 9728, "rank": 1},
    {"item_id": 15297, "rank": 2},
    {"item_id": 10440, "rank": 3},
    {"item_id": 13865, "rank": 4},
    {"item_id": 12360, "rank": 5}
  ],
  "model_type": "popular"
}
```

---

## CI/CD

**GitHub Actions** (`.github/workflows/ci.yml`):

```
On push/PR → flake8 → pytest → Build Docker → Push to ghcr.io
```

| Этап | Инструмент |
|------|-----------|
| Линтер | flake8 (0 errors) |
| Тесты | pytest |
| Сборка | Docker Buildx с layer caching |
| Регистрация | ghcr.io (SHA + branch + latest теги) |

---

## Лицензия

MIT
