import os
import csv
import chromadb
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from shared.core.config import settings
from pydantic import BaseModel # Keep pydantic if other parts of your app use it, otherwise optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader, ImageCaptionLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ddgs import DDGS
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import threading

from utils.model_loader import load_embedding_model

class RAGService:
    def __init__(self):
        print("🤖 GroundTruth AI Service: Initializing with Vector Database...")
        
        # 1. Initialize the embedding model from settings to convert text to vectors
        print(f" modeli: {settings.EMBEDDING_MODEL_NAME}")
        # self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        self.embedding_model = load_embedding_model(settings.EMBEDDING_MODEL_NAME, settings.HF_HOME)
        
        # 2. Initialize the ChromaDB client for vector storage
        # This data will be saved in the '/chroma_db' volume you created
        self.chroma_client = chromadb.PersistentClient(path="/chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="agri_knowledge_base")
        self.is_ready = threading.Event() # Flag to indicate when processing is done

        # Launch the heavy loading process in a background thread
        print("🚀 Launching background thread for document processing...")
        background_thread = threading.Thread(target=self._load_and_embed_data)
        background_thread.daemon = True
        background_thread.start()

        print("✅ GroundTruth AI Service: API is ready. Document loading in background.")

    def _load_and_embed_data(self):
        """Loads data from CSV and embeds it into ChromaDB if the collection is empty."""
        print(f"🧠 [BG Thread] Checking for new documents in /app/knowledge_documents...")
        
        # Correctly get the list of already processed source files
        existing_docs = self.collection.get(include=["metadatas"])
        processed_sources = set(meta.get('source') for meta in existing_docs['metadatas'] if meta and meta.get('source'))
        
        print(f"📚 Found {len(processed_sources)} documents already processed.")

        doc_directory = './knowledge_documents/'

        #Create a list of all files to process first for the progress bar ---
        files_to_process = []

        # Iterate through all files in the directory
        for dirpath, _, filenames in os.walk(doc_directory):
            for filename in filenames:
                # Skip the metadata.json file explicitly
                if filename == "metadata.json":
                    continue

                file_path = os.path.join(dirpath, filename)
                if file_path not in processed_sources:
                    files_to_process.append(file_path)
                
        if not files_to_process:
            print("✅ No new documents to add. Knowledge base is up to date.")
            # When done, set the ready flag
            self.is_ready.set()
            print("✅ [BG Thread] Document processing complete. Service is fully operational.")
            return
                
        total_files = len(files_to_process)
        print(f"⏳ Found {total_files} new files to process.")

        #Process files one by one with progress logging
        for i, file_path in enumerate(files_to_process):
            filename = os.path.basename(file_path)
            print(f"\n[{i+1}/{total_files}] Processing: {filename}...")

            try:
                print(f"⏳ Attempting to load {file_path}...")
                doc_to_process = []
                file_extension = os.path.splitext(filename)[1].lower()

                if file_extension == ".pdf":
                    combined_text = ""
                    loader = PyPDFLoader(file_path)
                    text_docs = loader.load()
                    if any(doc.page_content.strip() for doc in text_docs if isinstance(doc, Document)):
                        combined_text += "\n".join([doc.page_content for doc in text_docs])
                    else:
                        # This handles the unexpected tuple case
                        print(f"⚠️ Found an unexpected data type in '{filename}': {type(item)}")
                        print(f"   Content of unexpected data: {item}")
                        print("   Skipping this part to prevent errors.")
                
                    #Always apply OCR to every page to get text from images
                    print(f"✨ Applying OCR to pages in '{filename}' to find text in images...")
                    images = convert_from_path(file_path)
                    for i, img in enumerate(images):
                        ocr_text = pytesseract.image_to_string(img)
                        if ocr_text.strip(): # Add OCR text if it's not empty
                            combined_text += f"\n--- OCR Text from Page {i+1} ---\n" + ocr_text
                
                    if combined_text:
                        doc_to_process = [Document(page_content=combined_text, metadata={'source': file_path})]
                elif file_extension == ".txt":
                    loader = TextLoader(file_path, encoding='utf-8')
                    doc_to_process = loader.load()
                elif file_extension in [".jpg", ".jpeg", ".png"]:
                    ocr_text = pytesseract.image_to_string(Image.open(file_path))
                    doc_to_process = [Document(page_content=ocr_text, metadata={'source': file_path})]

                if not doc_to_process:
                    print(f"⚠️ No content extracted from {filename}. Skipping.")
                    continue
                
                print(f"✅ Successfully loaded {filename}")

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = text_splitter.split_documents(doc_to_process)
                
                chunk_texts = [doc.page_content for doc in docs]
                metadatas = [{'source': doc.metadata['source']} for doc in docs]
                embeddings = self.embedding_model.encode(chunk_texts).tolist()
                ids = [f"{doc.metadata['source']}_chunk_{i}" for i, doc in enumerate(docs)]
                
                self.collection.add(
                    embeddings=embeddings,
                    documents=chunk_texts,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"✅ Successfully embedded and stored {filename} ({len(docs)} chunks.")

            except Exception as e:
                # This will now catch and report the error for the specific file
                print(f"❌ ERROR loading/Could not embed new documents for '{filename}': {e}. Skipping file.")
        
        # When done, set the ready flag
        self.is_ready.set()
        print("✅ [BG Thread] Document processing complete. Service is fully operational.")

    def _web_search(self, query: str):
        """Performs a web search using DuckDuckGo if internal knowledge fails."""
        print(f"💻 No relevant context found internally. Performing web search for: '{query}'")
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=3)]
            
            if not results:
                return "No results found from web search."

            # Format the results into a single string
            search_context = "Web search results:\n"
            for result in results:
                search_context += f"- {result['title']}: {result['body']} (Source: {result['href']})\n"
            return search_context
        except Exception as e:
            print(f"❌ Web search failed: {e}")
            return "Could not perform web search."

    def _find_best_match(self, query: str) -> str | None:
        """Performs a semantic search in ChromaDB to find the best context."""
        if self.collection.count() == 0:
            print("⚠️ Collection count is 0. Switching to web search.")
            context = self._web_search(query)
            return context
            # return "Knowledge base is empty."

        # 1. Convert the user's query into a vector embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # 2. Query the vector database to find similar document
        # Combine the content of the top 2 chunks for richer context
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
            include=["documents", "embeddings"]
        )
        
        # 3. Extract the answer
        if results and results['documents'] and results['documents'][0]:

            doc_embedding = results['embeddings'][0][0]
            # Calculate cosine similarity
            similarity = dot(query_embedding, doc_embedding) / (norm(query_embedding) * norm(doc_embedding))
            print(f"📊 Semantic similarity score: {similarity:.4f}")

            # If similarity is high, use the internal document
            if similarity > 0.4: # You can tune this threshold
                print("✅ Found relevant context in internal documents.")
                context = results['documents'][0][0]
            else:
                # If not relevant enough, use the web search tool
                print("⚠️ Local context not relevant enough. Switching to web search.")
                context = self._web_search(query)
            return context
        else:
            # If no documents are found at all, use the web search tool
            print("⚠️ No documents are found at all. Switching to web search.")
            context = self._web_search(query)


            return context
            
        return None
        # return "No relevant context found in the knowledge base."

    async def get_response(self, query: str) -> str:
        """Performs RAG using the vector database and a local LLM."""
        if not self.is_ready.is_set():
            return "The knowledge base is still being built. Please try again in a moment."
            

        from litellm import acompletion # Using lazy import for stability
        
        print(f"🔎 Performing semantic search for: '{query}'")
        context = self._find_best_match(query)

        if not context:
            print("🤷 No relevant context found in knowledge base/vector database/websearch.")
            context = "No specific information available."

        print(f"📝 Context found: '{context}'")
        
        system_prompt = """You are a factual database assistant for agriculture. Your task is to answer the user's question exclusively based on the provided text context. Follow these rules strictly:
        1. Analyze the context to see if it directly answers the user's question.
        2. If it directly answers the question, provide that answer clearly based ONLY on that context.
        3. If the context is related but does not directly answer the question, summarize what the context says and then explicitly state that a direct answer is not in the document.
        4. If the context is NOT relevant, you MUST state that you do not have enough information on that specific topic, and do not use the irrelevant context.
        5. If the context is from a web search, you MUST cite the source URL provided for each piece of information.
        6. CRITICAL RULE: Do NOT use any external knowledge. Do NOT add any information that is not explicitly present in the provided context."""

        user_prompt = f"Context: \"{context}\"\n\nQuestion: \"{query}\"\n\nAnswer:"
        
        try:
            print(f"🤖 Sending request to local LLM: {settings.LOCAL_LLM_MODEL}")
            response = await acompletion(
                model=f"ollama/{settings.LOCAL_LLM_MODEL}",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                api_base=settings.OLLAMA_API_BASE_URL
            )
            ai_response = response.choices[0].message.content
            print(f"💬 LLM Response: {ai_response}")
            return ai_response
        except Exception as e:
            print(f"❌ ERROR connecting to local LLM: {e}")
            return "Sorry, the AI assistant is currently unavailable."

rag_service = RAGService()