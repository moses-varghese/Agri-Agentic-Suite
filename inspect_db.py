import chromadb
import pandas as pd

# This script connects to the running ChromaDB instance inside the Docker volume.
client = chromadb.PersistentClient(path="./chroma_db_volume") # Assumes a local mount for inspection

def inspect_collection(collection_name):
    try:
        collection = client.get_collection(name=collection_name)
        data = collection.get(include=["metadatas", "documents"])

        if not data or not data['ids']:
            print(f"Collection '{collection_name}' is empty or does not exist.")
            return

        df = pd.DataFrame({
            'id': data['ids'],
            'document_chunk': data['documents'],
            'source': [meta.get('source', 'N/A') for meta in data['metadatas']]
        })

        print(f"\n--- Contents of '{collection_name}' Collection ({len(df)} chunks) ---")
        print(df.to_markdown(index=False))

    except Exception as e:
        print(f"Error inspecting collection '{collection_name}': {e}")


if __name__ == "__main__":
    print("Inspecting ChromaDB...")
    # Add a local volume mount to your docker-compose.yml for this script to work
    # under groundtruth_ai service:
    # volumes:
    #   - ./chroma_db_volume:/chroma_db
    inspect_collection("agri_knowledge_base")