import os
import json
import requests, cloudscraper, certifi
import shutil

# Paths are relative to the container's working directory (/app)
METADATA_PATH = 'knowledge_documents/metadata.json'
DOCS_FOLDER = 'knowledge_documents'

cafile = certifi.where()
# Step 1: Copy certifi bundle into a new file
shutil.copyfile(cafile, "combined.pem")
with open("govt.pem", "rb") as infile, open("combined.pem", "ab") as outfile:
    outfile.write(infile.read())

def download_knowledge_base():
    print("🚀 Starting knowledge base synchronization...")

    if not os.path.exists(METADATA_PATH):
        print(f"⚠️ WARNING: Cannot find '{METADATA_PATH}'. Skipping document download.")
        return

    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    os.makedirs(DOCS_FOLDER, exist_ok=True)

    for item in metadata:
        file_name = f"{item['id']}.pdf"
        file_path = os.path.join(DOCS_FOLDER, file_name)

        if os.path.exists(file_path):
            print(f"✅ SKIPPING: '{file_name}' already exists.")
            continue

        print(f"⏳ DOWNLOADING: '{item['title']}'...")
        try:
            response = requests.get(item['download_url'], headers = item.get('headers'), allow_redirects=True, verify = 'combined.pem', stream=True)#verify = False less secure
            response.raise_for_status()
            if response.status_code == 403:
                scraper = cloudscraper.create_scraper()
                r = scraper.get(item['download_url'], headers = item.get('headers'), allow_redirects=True, verify = 'combined.pem', timeout=20) #verify = False less secure
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✅ SUCCESS by cloudscraper: Saved to '{file_path}'")
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ SUCCESS: Saved to '{file_path}'")
        except requests.exceptions.RequestException as e:
            print(f"❌ FAILED to download '{item['title']}'. Error: {e}")

    print("\n✨ Knowledge base synchronization complete.")

if __name__ == "__main__":
    download_knowledge_base()