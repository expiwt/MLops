# Kion RecSys — MLOps Project

Рекомендательная система для онлайн-кинотеатра Kion с полным MLOps-циклом:
от версионирования данных до мониторинга в Kubernetes.

## Содержание

- [Архитектура](#архитектура)
- [Состав проекта](#состав-проекта)
- [Быстрый старт](#быстрый-старт)
- [Работа с Kubernetes](#работа-с-kubernetes)
- [Модели и MLflow](#модели-и-mlflow)
- [Web UI](#web-ui)
- [API Endpoints](#api-endpoints)
- [CI/CD](#cicd)
- [Покрытие требований курса](#покрытие-требований-курса)

---

## Архитектура

```
┌──────────────────────────────────────────────────────────┐
│                    GitHub Actions                         │
│       Lint → Test → Build Docker → Push to Registry      │
└───────────────────────┬──────────────────────────────────┘
                        │ CD (ArgoCD sync)
                        ▼
┌──────────────────────────────────────────────────────────┐
│                    Kubernetes (kind)                      │
│                                                           │
│  ┌─────────────┐  ┌──────────┐  ┌──────────┐            │
│  │  FastAPI     │  │  MLflow  │  │Portainer │            │
│  │  RecSys API  │  │ Tracking │  │ K8s UI   │            │
│  │  :8000      │  │ :5000   │  │ :9000   │            │
│  └──────┬───────┘  └──────────┘  └──────────┘            │
│         │                                                 │
│  └──────────────┘                                        │
└──────────────────────────────────────────────────────────┘

                 ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
                  Локально (docker-compose)
                 │  + Prometheus :9090       │
                  + Grafana    :3000
                 └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
```

**Компоненты:**

| Компонент | Назначение |
|-----------|-----------|
| **FastAPI сервис** | Выдаёт рекомендации, Web UI |
| **MLflow** | Трекинг экспериментов, регистрация моделей |
| **Prometheus / Grafana** | Сбор и визуализация метрик (RPS, latency) |
| **Portainer** | Веб-интерфейс управления Kubernetes |
| **Kind** | Локальный Kubernetes-кластер в Docker |
| **DVC** | Версионирование данных и моделей |

---

## Состав проекта

```
├── app/                        # ★ FastAPI сервис
│   ├── main.py                 #  Эндпоинты, Prometheus
│   └── templates/index.html    #  Web UI (инференс, история)
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
│   ├── network-policy.yaml     #  Сетевая изоляция
│   ├── resource-quota.yaml     #  Квоты
│   ├── argocd/                 #  ArgoCD Application manifest
│   ├── prometheus/             #  ServiceMonitor для Prometheus Operator
│   └── portainer/              #  Portainer (k8s web UI)
│
├── infra/                      # ★ Мониторинг (для docker-compose)
│   ├── prometheus/prometheus.yml
│   └── grafana/
│       ├── provisioning/       #  Автоконфиг датасорсов и дашбордов
│       └── dashboards/         #  JSON-дашборды
│
├── tests/                      # ★ Тесты
│   └── test_predict.py         #  Тесты инференса
│
├── data/                       # Данные под DVC
├── models/                     # Обученные модели под DVC
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

# Развернуть
make k8s-apply
kubectl apply -f k8s/deployment.yaml
kubectl get pods -n recsys -w

kubectl port-forward -n recsys svc/recsys-api 8000:8000
kubectl port-forward -n recsys svc/mlflow 5000:5000
kubectl port-forward -n portainer svc/portainer 9000:9000


# Тест API
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict/123?top_k=5&model_type=popular"
```

### Что развёрнуто

| Компонент | Образ | Ресурсы |
|-----------|-------|---------|
| recsys-api | mlops-recsys-api:latest | 256Mi / 1Gi |
| mlflow | ghcr.io/mlflow/mlflow:v2.11.3 | 128Mi / 512Mi | 
| portainer | portainer/portainer-ce:latest | 128Mi / 512Mi | 

### ArgoCD

```bash
kubectl apply -f k8s/argocd/application.yaml
```

Также возможно развернуть на **Minikube** или любом другом кластере —
манифесты универсальны.

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

Модели для инференса загружаются напрямую из `models/*.pkl`, а не через MLflow Registry.

### Переобучение

```bash
# Через Web UI — кнопка "Запустить переобучение"
# Через API:
curl -X POST http://localhost:8000/retrain
curl http://localhost:8000/retrain/status
```

---

## Web UI

Доступен по http://localhost:8000:

- **Форма инференса** — ввод user_id, количества рекомендаций, выбор модели


- **Статус системы** — загруженные модели
- **Кнопка переобучения** — запуск retrain
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

## Покрытие требований курса

| № | Требование | Статус |
|---|-----------|--------|
| 1 | Датасет, базовая модель | + Kion + PopularRecommender + TFIDFRecommender |
| 2 | Git + conventional commits + DVC | + |
| 3 | Cookiecutter-шаблон | + |
| 4 | MLflow трекинг и регистрация | + |
| 5 | CI/CD (линтер, тесты, сборка, деплой) | + GitHub Actions |
| 6 | FastAPI + OpenAPI + Docker + Kubernetes | + Kind с полным набором манифестов |
| 7 | Prometheus + Grafana | + мониторинг |
| 8 | — | — |
| 9 | Web UI | + Инференс, история, retrain |
| 10 | ArgoCD | + Application manifest |
| 11 | README | + |

---

## Лицензия

MIT