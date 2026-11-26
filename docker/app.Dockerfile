# Image for the API/dashboard and the Kafka sentiment consumer.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NLTK_DATA=/usr/share/nltk_data
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/cryptosent
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[mlflow]" && mkdir -p /opt/cryptosent/artifacts && chmod 777 /opt/cryptosent/artifacts
COPY docker/nltk_setup.py /tmp/nltk_setup.py
RUN python /tmp/nltk_setup.py

EXPOSE 8000
CMD ["cryptosent", "serve", "--host", "0.0.0.0", "--port", "8000"]
