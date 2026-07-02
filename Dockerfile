FROM python:3.12-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b

RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/sh --create-home appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY taxi_driver/ taxi_driver/
COPY utils/ utils/
COPY sarsa/ sarsa/

RUN chown -R appuser:appgroup /app

USER appuser

WORKDIR /app/sarsa

ENV HEADLESS=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLBACKEND=Agg
ENV REFRESH_EVERY=250

CMD ["python", "-u", "main.py"]
