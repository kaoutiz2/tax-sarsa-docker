FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends xvfb xauth \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY taxi_driver/ taxi_driver/
COPY utils/ utils/
COPY sarsa/ sarsa/

WORKDIR /app/sarsa

ENV HEADLESS=1
ENV PYTHONUNBUFFERED=1

CMD ["xvfb-run", "-a", "python", "main.py"]
