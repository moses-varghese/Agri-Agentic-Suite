#!/bin/sh
# This script starts the Ollama server and pulls a list of models.

/bin/ollama serve &
pid=$!
echo "Ollama server started with PID: $pid"
sleep 5 # Give the server a moment to start

# Loop through all model names passed as arguments
for model in "$@"; do
  echo "Checking for model: $model"
  if ollama list | grep -q "$model"; then
    echo "✅ Model '$model' already exists."
  else
    echo "⏳ Model '$model' not found. Pulling..."
    ollama pull "$model"
  fi
done

echo "Ollama is ready. Initial model setup complete."
wait $pid