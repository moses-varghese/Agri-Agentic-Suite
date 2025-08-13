# from fastapi import UploadFile
# import random

# class VisionService:
#     @staticmethod
#     async def analyze_image(image_file: UploadFile) -> dict:
#         """
#         MVP Vision Service.
#         Later, this will use a TFLite model for inference.
#         For now, it returns a random dummy diagnosis.
#         """
#         print(f"✅ FieldScout AI: Vision service received image: '{image_file.filename}'")
#         possible_diseases = ["Leaf Blight", "Powdery Mildew", "Rust", "Healthy"]
#         diagnosis = random.choice(possible_diseases)
#         confidence = random.uniform(0.75, 0.98)
        
#         return {
#             "filename": image_file.filename,
#             "content_type": image_file.content_type,
#             "diagnosis": diagnosis,
#             "confidence": f"{confidence:.2f}",
#             "recommendation": f"This is a dummy recommendation for {diagnosis}."
#         }

# vision_service = VisionService()

import base64
from fastapi import UploadFile
from shared.core.config import settings

class VisionService:
    @staticmethod
    async def analyze_image(image_file: UploadFile) -> dict:
        """
        Analyzes an image using the LLaVA multimodal model.
        """
        # Lazy import litellm for stability, as we've learned.
        from litellm import acompletion

        try:
            # 1. Read the image content from the uploaded file.
            image_bytes = await image_file.read()
            
            # 2. Encode the image in base64, which is how LLaVA expects it.
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            # 3. Create the prompt for the vision model.
            prompt = "Analyze this image of a plant leaf. Identify any visible diseases or pests. Provide a brief diagnosis and a suggested course of action for a farmer. If the leaf appears healthy, state that."

            # 4. Send the request to the LLaVA model.
            print(f"🌿 Sending image to LLaVA for analysis...")
            response = await acompletion(
                model=f"ollama/llava",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }],
                api_base=settings.OLLAMA_API_BASE_URL
            )
            
            ai_response = response.choices[0].message.content
            print(f"💬 LLaVA Response: {ai_response}")

            # 5. Format the response for the frontend.
            return {
                "diagnosis": ai_response,
                "confidence": "High (AI Generated)",
                "recommendation": "Please follow the advice provided in the diagnosis."
            }

        except Exception as e:
            print(f"❌ ERROR in Vision Service: {e}")
            return {
                "diagnosis": "Error analyzing image.",
                "confidence": "N/A",
                "recommendation": "The AI vision model could not be reached. Please try again."
            }


vision_service = VisionService()