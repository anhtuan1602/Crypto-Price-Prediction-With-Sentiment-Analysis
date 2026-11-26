# Airflow with the cryptosent package installed, so DAG tasks can import it.
FROM apache/airflow:2.10.3-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
ENV NLTK_DATA=/usr/share/nltk_data
COPY docker/nltk_setup.py /tmp/nltk_setup.py
COPY pyproject.toml README.md /opt/cryptosent/
COPY src /opt/cryptosent/src
RUN mkdir -p /opt/cryptosent/artifacts && chown -R airflow: /opt/cryptosent && mkdir -p /usr/share/nltk_data && chown -R airflow: /usr/share/nltk_data

USER airflow
RUN pip install --no-cache-dir "/opt/cryptosent[mlflow]" && python /tmp/nltk_setup.py
