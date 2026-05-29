FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY taxi_driver/ taxi_driver/
COPY utils/ utils/
COPY sarsa/ sarsa/

WORKDIR /app/sarsa

ENV HEADLESS=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg
ENV REFRESH_EVERY=250

CMD ["python", "-u", "main.py"]
