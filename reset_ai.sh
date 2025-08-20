#!/bin/bash
echo "Stopping all services..."
docker-compose down

echo "Removing Ollama and ChromaDB volumes..."
docker volume rm agri-agentic-suite_ollama_data
docker volume rm agri-agentic-suite_chroma_db

echo "✅ AI data has been reset. Your PostgreSQL database is untouched."