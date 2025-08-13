import json
from litellm import acompletion
from shared.core.config import settings

class FinancialAgent:
    def __init__(self, data_path: str = "/app/local_data/schemes.json"):
        print(f"🧠 Loading financial schemes from: {data_path}")
        try:
            with open(data_path, 'r') as f:
                self.schemes = json.load(f)
            print("✅ Financial schemes loaded successfully.")
        except FileNotFoundError:
            print(f"❌ ERROR: Schemes file not found at {data_path}")
            self.schemes = []

    def _find_eligible_schemes(self, land_size: float) -> list:
        """Finds eligible schemes based on simple logic."""
        eligible = []
        for scheme in self.schemes:
            if "landholding" in scheme.get("eligibility", "").lower():
                if land_size < 5: # Simple dummy logic
                    eligible.append(scheme)
            else:
                 eligible.append(scheme) # Assume eligible if no landholding rule
        return eligible

    async def get_financial_plan(self, land_size: float, crop: str) -> dict:
        """
        Generates a financial plan using local data and the local LLM.
        """
        print(f"🔎 Finding schemes for {land_size} acres.")
        eligible_schemes = self._find_eligible_schemes(land_size)

        if not eligible_schemes:
            return {"error": "No eligible schemes found for your profile."}
        
        context = f"A farmer with {land_size} acres wants to plant {crop}. The following government schemes are available to them: {json.dumps(eligible_schemes)}."
        
        system_prompt = "You are a helpful agricultural finance assistant. Based ONLY on the provided context, create a brief, bulleted financial plan. Mention the eligible schemes and suggest what the farmer should consider."
        
        user_prompt = f"Context: \"{context}\"\n\nCreate a simple financial plan."

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
            ai_plan = response.choices[0].message.content
            print(f"💬 LLM Response: {ai_plan}")
            return {"plan": ai_plan}
        except Exception as e:
            print(f"❌ ERROR connecting to local LLM: {e}")
            return {"error": "Sorry, the AI assistant is currently unavailable."}

financial_agent = FinancialAgent()