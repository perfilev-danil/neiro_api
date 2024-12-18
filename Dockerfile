# Используем официальный образ Python в качестве базового
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем файлы requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы приложения
COPY . .

ARG CA
ENV CA=${CA}

ARG PASS
ENV PASS=${PASS}

ARG HOSTS
ENV HOSTS=${HOSTS}

ARG SOURCE_DIR
ENV SOURCE_DIR=${SOURCE_DIR}

ARG CHUNK_SIZE
ENV CHUNK_SIZE=${CHUNK_SIZE}

ARG CHUNK_OVERLAP
ENV CHUNK_OVERLAP=${CHUNK_OVERLAP}

ARG SERVICE_ACCOUNT_ID
ENV SERVICE_ACCOUNT_ID=${SERVICE_ACCOUNT_ID}

ARG KEY_ID
ENV KEY_ID=${KEY_ID}

ARG PRIVATE
ENV PRIVATE=${PRIVATE}

# Указываем команду для запуска приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
