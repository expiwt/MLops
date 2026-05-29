FROM python:3.12-slim

WORKDIR /app

# Установка системных библиотек (libgomp нужен для implicit/TF-IDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости + setup.py (для -e . в requirements.txt)
COPY requirements.txt setup.py .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Копируем исходный код
COPY src/ src/
COPY app/ app/

# Копируем модели и фичи (для прода — через DVC)
COPY models/ models/
COPY data/features/ data/features/

# Healthcheck — проверяет, что сервис жив
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
