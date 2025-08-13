# class RAGService:
#     @staticmethod
#     async def get_response(query: str) -> str:
#         """
#         MVP RAG Service.
#         Later, this will perform embedding, vector search, and LLM generation.
#         For now, it returns a hardcoded response based on keywords.
#         """
#         print(f"✅ GroundTruth AI: RAG service received query: '{query}'")
#         query_lower = query.lower()
#         if "wheat" in query_lower and "rust" in query_lower:
#             return "For wheat rust, consider using a fungicide like Propiconazole. Check with your local agri-advisor for specific dosage."
#         elif "irrigate" in query_lower:
#             return "Check soil moisture before irrigating. Generally, irrigate early in the morning to reduce evaporation."
#         else:
#             return "Thank you for your query. We are processing it and will get back to you shortly."

# rag_service = RAGService()


import pandas as pd
from litellm import acompletion
from shared.core.config import settings
from pydantic import BaseModel

class RAGService:
    # def __init__(self, data_path: str = "local_data/knowledge_base.csv"):
    def __init__(self, data_path: str = "/app/local_data/knowledge_base.csv"):
        print(f"🧠 Loading knowledge base from: {data_path}")
        try:
            self.knowledge_base = pd.read_csv(data_path)
            print("✅ Knowledge base loaded successfully.")
        except FileNotFoundError:
            print(f"❌ ERROR: Knowledge base file not found at {data_path}")
            self.knowledge_base = pd.DataFrame(columns=['question', 'answer'])

    def _find_best_match(self, query: str) -> str | None:
        """
        A simple keyword-based search to find the most relevant document.
        In a real system, this would use vector embeddings.
        """
        if self.knowledge_base.empty:
            return None
        
        query_words = set(query.lower().split())
        
        # Calculate a simple match score for each question
        self.knowledge_base['score'] = self.knowledge_base['question'].apply(
            lambda q: len(set(q.lower().split()) & query_words)
        )
        
        best_match = self.knowledge_base.sort_values(by='score', ascending=False).iloc[0]
        
        if best_match['score'] > 0:
            return best_match['answer']
        return None

    async def get_response(self, query: str) -> str:
        """
        Performs RAG using the local knowledge base and local LLM.
        """
        print(f"🔎 Finding context for query: '{query}'")
        context = self._find_best_match(query)

        if not context:
            print("🤷 No relevant context found in knowledge base.")
            context = "No specific information available."

        print(f"📝 Context found: '{context}'")
        
        system_prompt = "You are a helpful agricultural assistant. Your answers must be concise, easy to understand for a farmer, and based ONLY on the provided context. Do not add any information not present in the context. If the context is not helpful, just say that you don't have enough information."
        
        user_prompt = f"Context: \"{context}\"\n\nQuestion: \"{query}\"\n\nAnswer:"
        
        try:
            print(f"🤖 Sending request to local LLM: {settings.LOCAL_LLM_MODEL}")
            response = await acompletion(
                model=f"ollama/{settings.LOCAL_LLM_MODEL}",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                api_base=settings.OLLAMA_API_BASE_URL,
                stream=False
            )
            ai_response = response.choices[0].message.content
            print(f"💬 LLM Response: {ai_response}")
            return ai_response
        except Exception as e:
            print(f"❌ ERROR connecting to local LLM: {e}")
            return "Sorry, the AI assistant is currently unavailable."


rag_service = RAGService()