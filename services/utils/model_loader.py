# utils/model_loader.py
from sentence_transformers import SentenceTransformer
import logging
import os

class ModelUnavailableError(Exception):
    pass

def load_embedding_model(model_name: str, cache_path: str):
    try:
        model = SentenceTransformer(model_name, cache_folder=cache_path)
        logging.info(f"✅ Loaded embedding model '{model_name}' "
                     f"(cache exists: {os.path.exists(cache_path)})")
        return model
    except OSError as e:
        logging.error(f"❌ Could not load model '{model_name}'. "
                      f"Cache missing and network unavailable. Error: {e}")
        raise ModelUnavailableError(
            f"No available model '{model_name}'. "
            f"Check network or cache at {cache_path}."
        )
