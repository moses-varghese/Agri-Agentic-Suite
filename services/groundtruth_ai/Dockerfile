# # Use an official Python runtime as a parent image
# FROM python:3.11-slim

# Use the full Python image for maximum compatibility
FROM python:3.11

# Install Tesseract OCR Engine and its dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    poppler-utils \
    ffmpeg \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and unzip the offline Vosk speech recognition model
RUN apt-get update && apt-get install -y wget unzip && \
    # Step 1: Download the file to a specific path
    wget -O /tmp/vosk.zip "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" && \
    # Step 2: Unzip the file from that path
    unzip /tmp/vosk.zip -d /opt && \
    mv /opt/vosk-model-small-en-us-0.15 /opt/vosk-model && \
    rm /tmp/vosk.zip && \
    apt-get purge -y wget unzip && apt-get autoremove -y

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./services/groundtruth_ai/requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the shared code and the service's source code into the container
COPY ./services/shared /app/shared
COPY ./services/utils /app/utils
COPY ./services/groundtruth_ai/app /app/app

# Copy the new scripts
COPY ./services/groundtruth_ai/download_docs.py /app/download_docs.py
COPY ./services/groundtruth_ai/entrypoint.sh /app/entrypoint.sh
COPY ./services/groundtruth_ai/govt.pem /app/govt.pem

# Make the entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]