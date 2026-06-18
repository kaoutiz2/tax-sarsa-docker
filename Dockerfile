FROM python:3.12-slim-bookworm

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
