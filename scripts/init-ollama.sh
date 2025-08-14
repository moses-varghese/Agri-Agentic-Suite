#!/bin/sh
# This script robustly pulls models first, then starts the main server.

# 1. Store the model names from the command arguments
MODELS_TO_PULL="$@"

# 2. Start a temporary server in the background to handle pull requests
/bin/ollama serve &
pid=$!
echo "Ollama server started temporarily in background (PID: $pid) for model pulling."
sleep 5 # Give the server a moment to start up

# 3. Loop through and pull all required models
for model in $MODELS_TO_PULL; do
  echo "Checking for model: $model"
  # Use the API to check for the model, it's more reliable
  if curl -s --fail http://localhost:11434/api/tags | grep -q "$model"; then
    echo "✅ Model '$model' already exists."
  else
    echo "⏳ Pulling model '$model'..."
    ollama pull "$model"
  fi
done

# 4. Stop the temporary server
echo "Model setup complete. Shutting down temporary server."
kill $pid
wait $pid 2>/dev/null # Wait for the process to terminate

# 5. Start the main server in the foreground
# This will now be the main process for the container.
echo "🚀 Starting main Ollama server..."
/bin/ollama serve