# This syntax enables the --mount=type=secret feature
#syntax=docker/dockerfile:1

FROM python:3.11

WORKDIR /app

COPY ./services/yieldwise/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./services/yieldwise/train_price_estimator.py /app/train_price_estimator.py

# --- THIS IS THE FIX ---
# Run the training script, securely mounting the .env file as a secret.
# Pydantic will automatically find and read the .env file at this location.
# RUN --mount=type=secret,id=.env,target=/app/.env python /app/train_price_estimator.py
# RUN --mount=type=secret,id=.env,target=/app/.env cat /app/.env

# Copy the main application code after the script has run
COPY ./services/shared /app/shared
COPY ./services/utils /app/utils
COPY ./services/yieldwise/app /app/app
