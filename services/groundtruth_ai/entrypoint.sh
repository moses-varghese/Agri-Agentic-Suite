#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Run the download script first to ensure documents are up to date.
echo "--- Running knowledge base sync script ---"
python download_docs.py
echo "--- Knowledge base sync complete ---"

# Then, execute the command passed to this script (e.g., the uvicorn server).
exec "$@"