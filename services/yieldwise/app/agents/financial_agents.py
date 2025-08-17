# import json
# from litellm import acompletion
# from shared.core.config import settings

# class FinancialAgent:
#     def __init__(self, data_path: str = "/app/local_data/schemes.json"):
#         print(f"🧠 Loading financial schemes from: {data_path}")
#         try:
#             with open(data_path, 'r') as f:
#                 self.schemes = json.load(f)
#             print("✅ Financial schemes loaded successfully.")
#         except FileNotFoundError:
#             print(f"❌ ERROR: Schemes file not found at {data_path}")
#             self.schemes = []

#     def _find_eligible_schemes(self, land_size: float) -> list:
#         """Finds eligible schemes based on simple logic."""
#         eligible = []
#         for scheme in self.schemes:
#             if "landholding" in scheme.get("eligibility", "").lower():
#                 if land_size < 5: # Simple dummy logic
#                     eligible.append(scheme)
#             else:
#                  eligible.append(scheme) # Assume eligible if no landholding rule
#         return eligible

#     async def get_financial_plan(self, land_size: float, crop: str) -> dict:
#         """
#         Generates a financial plan using local data and the local LLM.
#         """
#         print(f"🔎 Finding schemes for {land_size} acres.")
#         eligible_schemes = self._find_eligible_schemes(land_size)

#         if not eligible_schemes:
#             return {"error": "No eligible schemes found for your profile."}
        
#         context = f"A farmer with {land_size} acres wants to plant {crop}. The following government schemes are available to them: {json.dumps(eligible_schemes)}."
        
#         system_prompt = "You are a helpful agricultural finance assistant. Based ONLY on the provided context, create a brief, bulleted financial plan. Mention the eligible schemes and suggest what the farmer should consider."
        
#         user_prompt = f"Context: \"{context}\"\n\nCreate a simple financial plan."

#         try:
#             print(f"🤖 Sending request to local LLM: {settings.LOCAL_LLM_MODEL}")
#             response = await acompletion(
#                 model=f"ollama/{settings.LOCAL_LLM_MODEL}",
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 api_base=settings.OLLAMA_API_BASE_URL,
#                 stream=False
#             )
#             ai_plan = response.choices[0].message.content
#             print(f"💬 LLM Response: {ai_plan}")
#             return {"plan": ai_plan}
#         except Exception as e:
#             print(f"❌ ERROR connecting to local LLM: {e}")
#             return {"error": "Sorry, the AI assistant is currently unavailable."}

# financial_agent = FinancialAgent()

import chromadb
from sentence_transformers import SentenceTransformer
from shared.core.config import settings
from ddgs import DDGS
from numpy import dot
from numpy.linalg import norm

class FinancialAgent:
    def __init__(self):
        print("💰 YieldWise Service: Initializing with RAG capabilities...")
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        
        # Connect to the same persistent ChromaDB that groundtruth_ai uses
        self.chroma_client = chromadb.PersistentClient(path="/chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="agri_knowledge_base")
        
        print("✅ YieldWise Service: Initialization complete.")

    
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

    def _find_relevant_financial_context(self, query: str):
        """
        Performs a semantic search in ChromaDB to find financial context.
        """
        if self.collection.count() == 0:
            print("No documents found in the knowledge base. Switching to web search")
            context = self._web_search(query)
            return context

        # Create an embedding for the user's financial situation
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Find the top 3 most relevant document chunks
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3, # Get more context for financial planning
            include=["documents", "embeddings"]
        )
        
        if results and results['documents'] and results['documents'][0]:
            doc_embedding = results['embeddings'][0][0]
            # Calculate cosine similarity
            similarity = dot(query_embedding, doc_embedding) / (norm(query_embedding) * norm(doc_embedding))
            print(f"📊 Semantic similarity score: {similarity:.4f}")
            # Combine the text from all retrieved chunks
            # return " ".join(results['documents'][0])

            # If similarity is high, use the internal document
            if similarity > 0.5: # You can tune this threshold
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

        print("No specific financial information found in the documents.")
        return None

    async def get_financial_plan(self, land_size: float, crop: str) -> dict:
        from litellm import acompletion
        
        # Create a detailed query describing the user's situation
        user_situation = f"A farmer with {land_size} acres of land planning to grow {crop} needs a financial plan. They are interested in loans, government schemes like the Kisan Credit Card, and income support."
        
        print(f"🔎 Performing semantic search and Searching for financial context for: '{crop} farming at {land_size} acres'")
        context = self._find_relevant_financial_context(user_situation)

        if not context:
            print("🤷 No relevant context found in knowledge base/vector database/websearch.")
            context = "No specific information available."

        print(f"📝 Context found: '{context}'")

        # A more sophisticated prompt for financial analysis
        system_prompt = """
        You are an expert agricultural financial advisor. Your task is to create a financial plan for a farmer based ONLY on the provided context.
        Follow these steps:
        1.  Analyze the user's situation (land size, crop).
        2.  Review the provided context from internal documents (which may contain details on schemes like KCC, PM-KISAN, etc.).
        3.  Synthesize the information into a clear, bulleted financial plan.
        4.  If the context mentions specific schemes, highlight how they might apply to the farmer.
        5.  If the context is from a web search, you MUST cite the source URL provided for each piece of information.
        6.  If the context is not specific enough, provide general advice based on the available text and explicitly state what information is missing.
        7.  If the context is NOT relevant, you MUST state that you do not have enough information on that specific topic, and do not use the irrelevant context.
        8.  CRITICAL RULE: Do not invent schemes or financial details not present in the context."""

        user_prompt = f"Context: \"{context}\"\n\nUser Situation: \"{user_situation}\"\n\nFinancial Plan:"
        
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
            ai_plan = response.choices[0].message.content
            print(f"💬 LLM Response: {ai_plan}")
            return {"plan": ai_plan}
        except Exception as e:
            print(f"❌ ERROR connecting to local LLM: {e}")
            return "Sorry, the AI assistant is currently unavailable"

financial_agent = FinancialAgent()



