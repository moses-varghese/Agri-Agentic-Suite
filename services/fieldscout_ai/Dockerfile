# # Use an official Python runtime as a parent image
# FROM python:3.11-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the requirements file into the container at /app
# # COPY ./requirements.txt /app/requirements.txt
# COPY ./services/fieldscout_ai/requirements.txt /app/requirements.txt

# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# # Copy the shared code and the service's source code into the container
# # COPY ../shared /app/shared
# # COPY ./app /app/app
# COPY ./services/shared /app/shared
# COPY ./services/fieldscout_ai/app /app/app

# # The command to run the application will be provided by docker-compose

# Use the full Python image for maximum compatibility
FROM python:3.11

WORKDIR /app
COPY ./services/fieldscout_ai/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
COPY ./services/shared /app/shared
COPY ./services/fieldscout_ai/app /app/app