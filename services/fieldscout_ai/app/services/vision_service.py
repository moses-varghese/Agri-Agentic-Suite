# import base64
# from fastapi import UploadFile
# from shared.core.config import settings

# class VisionService:
#     @staticmethod
#     async def analyze_image(image_file: UploadFile) -> dict:
#         """
#         Analyzes an image using the LLaVA multimodal model.
#         """
#         # Lazy import litellm for stability, as we've learned.
#         from litellm import acompletion

#         try:
#             # 1. Read the image content from the uploaded file.
#             image_bytes = await image_file.read()
            
#             # 2. Encode the image in base64, which is how LLaVA expects it.
#             base64_image = base64.b64encode(image_bytes).decode("utf-8")

#             # 3. Create the prompt for the vision model.
#             prompt = "Analyze this image of a plant leaf. Identify any visible diseases or pests. Provide a brief diagnosis and a suggested course of action for a farmer. If the leaf appears healthy, state that."

#             # 4. Send the request to the LLaVA model.
#             print(f"🌿 Sending image to LLaVA for analysis...")
#             response = await acompletion(
#                 model=f"ollama/llava",
#                 messages=[{
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": prompt},
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/jpeg;base64,{base64_image}"
#                             }
#                         }
#                     ]
#                 }],
#                 api_base=settings.OLLAMA_API_BASE_URL
#             )
            
#             ai_response = response.choices[0].message.content
#             print(f"💬 LLaVA Response: {ai_response}")

#             # 5. Format the response for the frontend.
#             return {
#                 "diagnosis": ai_response,
#                 "confidence": "High (AI Generated)",
#                 "recommendation": "Please follow the advice provided in the diagnosis."
#             }

#         except Exception as e:
#             print(f"❌ ERROR in Vision Service: {e}")
#             return {
#                 "diagnosis": "Error analyzing image.",
#                 "confidence": "N/A",
#                 "recommendation": "The AI vision model could not be reached. Please try again."
#             }


# vision_service = VisionService()

# import base64
# import torch
# import os
# import io
# import timm
# from PIL import Image
# from fastapi import UploadFile
# from shared.core.config import settings
# from timm.data import resolve_data_config
# from timm.data.transforms_factory import create_transform

# class VisionService:
#     def __init__(self):
#         print("🌿 FieldScout AI: Initializing dual-model pipeline...")
#         # 1. Initialize the image classification model for confidence scores
#         # 'mobilenetv3_large_100' is lightweight and fast
#         self.classifier = timm.create_model('mobilenetv3_large_100', pretrained=True)
#         self.classifier.eval()
        
#         # Get the specific transformations required for this model
#         config = resolve_data_config({}, model=self.classifier)
#         self.transform = create_transform(**config)
        
#         # Load the class names the model was trained on
#         url, _ = timm.models.hf_hub_download_url(self.classifier)
#         with open(os.path.join(os.path.dirname(url), 'imagenet_classes.txt')) as f:
#             self.class_labels = [line.strip() for line in f.readlines()]
            
#         print("✅ Classification model loaded.")
#         print("✅ FieldScout AI Service is ready.")

#     def _get_confidence_score(self, image_bytes: bytes) -> (str, float):
#         """Uses the classification model to get a label and confidence score."""
#         try:
#             img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#             tensor = self.transform(img).unsqueeze(0) # Prepare image for the model
            
#             with torch.no_grad():
#                 output = self.classifier(tensor)
#                 probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
#             top_prob, top_catid = torch.topk(probabilities, 1)
            
#             confidence = top_prob.item()
#             label = self.class_labels[top_catid.item()]
            
#             print(f"📊 Classifier Result: '{label}' with confidence {confidence:.4f}")
#             return label, confidence
#         except Exception as e:
#             print(f"❌ ERROR in classifier: {e}")
#             return "N/A", 0.0

#     async def _get_descriptive_diagnosis(self, image_bytes: bytes) -> str:
#         """Uses the LLaVA model to get a descriptive diagnosis."""
#         from litellm import acompletion
#         base64_image = base64.b64encode(image_bytes).decode("utf-8")
#         prompt = "Analyze this image of a plant leaf. Concisely identify the most likely disease or pest. Provide a brief, one-sentence diagnosis."
        
#         try:
#             response = await acompletion(
#                 model=f"ollama/{settings.VISION_MODEL}",
#                 messages=[{"role": "user", "content": [
#                     {"type": "text", "text": prompt},
#                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
#                 ]}],
#                 api_base=settings.OLLAMA_API_BASE_URL)
#             return response.choices[0].message.content
#         except Exception as e:
#             print(f"❌ ERROR in LLaVA: {e}")
#             return "Could not get a descriptive diagnosis from the AI model."

#     async def analyze_image(self, image_file: UploadFile) -> dict:
#         image_bytes = await image_file.read()

#         # Run both models
#         descriptive_diagnosis = await self._get_descriptive_diagnosis(image_bytes)
#         _, confidence_score = self._get_confidence_score(image_bytes)
        
#         # Format the final response
#         return {
#             "diagnosis": descriptive_diagnosis,
#             "confidence": f"{confidence_score:.2%}", # Format as a percentage
#             "recommendation": "Consult with a local agricultural expert to confirm the AI-generated diagnosis."
#         }

# vision_service = VisionService()

# import base64
# import io
# import torch
# from PIL import Image
# from fastapi import UploadFile
# from shared.core.config import settings
# from transformers import AutoImageProcessor, BeitForImageClassification

# class VisionService:
#     def __init__(self):
#         print("🌿 FieldScout AI: Initializing dual-model pipeline...")
        
#         # 1. Initialize the specialized Plant Disease classification model
#         model_name = "microsoft/beit-base-patch16-224-pt22k-ft22k"
#         self.classifier_processor = AutoImageProcessor.from_pretrained(model_name)
#         self.classifier_model = BeitForImageClassification.from_pretrained(model_name)
        
#         print("✅ Classification model loaded.")
#         print("✅ FieldScout AI Service is ready.")

#     def _get_classification(self, image_bytes: bytes) -> (str, float):
#         """Uses the BEiT model to get a specific disease class and confidence score."""
#         try:
#             image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#             inputs = self.classifier_processor(images=image, return_tensors="pt")

#             with torch.no_grad():
#                 outputs = self.classifier_model(**inputs)
#                 logits = outputs.logits
            
#             # Get the top prediction
#             predicted_class_idx = logits.argmax(-1).item()
#             predicted_class = self.classifier_model.config.id2label[predicted_class_idx]

#             # Calculate confidence score
#             probabilities = torch.nn.functional.softmax(logits, dim=-1)
#             confidence = probabilities[0][predicted_class_idx].item()
            
#             print(f"📊 Classifier Result: '{predicted_class}' with confidence {confidence:.4f}")
#             return predicted_class, confidence
#         except Exception as e:
#             print(f"❌ ERROR in classifier: {e}")
#             return "Classification failed", 0.0

#     async def _get_descriptive_diagnosis(self, image_bytes: bytes) -> str:
#         """Uses the LLaVA model to get a human-readable description."""
#         from litellm import acompletion
#         base64_image = base64.b64encode(image_bytes).decode("utf-8")
#         prompt = """
#         You are an agricultural agronomy expert.
#         Identify the main subject (e.g., crop type, soil condition, pest, equipment, field view) and your task is to analyze this image of 
#         a plant for signs of disease or pest infestation.
#         1.  **Analyze:** Look for discoloration, spots, holes, wilting, or evidence of pests.
#         2.  **Diagnose:** State the most likely health issue. If the plant appears healthy, state "The plant appears to be healthy."
#         3.  **Recommend:** Provide a brief, actionable recommendation for a farmer.
#         4.   If the context is NOT relevant, you MUST state that you do not have enough information on that specific topic, and do not use the irrelevant context.
#         5.  **CRITICAL RULE:** Do NOT use any external knowledge. Do NOT add any information that is not explicitly present in the provided context.
#         Provide the output in the following format:
#         **Diagnosis:** [Your diagnosis]
#         **Confidence:** [High/Medium/Low]
#         **Recommendation:** [Your recommendation]
#         """

#         try:
#             print(f"🤖 Sending request to Vision Model: {settings.VISION_MODEL}")
#             response = await acompletion(
#                 model=f"ollama/{settings.VISION_MODEL}",
#                 messages=[{"role": "user", "content": [
#                     {"type": "text", "text": prompt},
#                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
#                 ]}],
#                 api_base=settings.OLLAMA_API_BASE_URL)
#             return response.choices[0].message.content
#         except Exception as e:
#             print(f"❌ ERROR in LLaVA: {e}")
#             return "Could not get a descriptive diagnosis from the AI model."

#     async def analyze_image(self, image_file: UploadFile) -> dict:
#         image_bytes = await image_file.read()

#         # Run both models in parallel
#         descriptive_diagnosis = await self._get_descriptive_diagnosis(image_bytes)
#         classification, confidence_score = self._get_classification(image_bytes)
        
#         # Format the final, combined response
#         return {
#             "classification": classification,
#             "confidence label of image": f"{confidence_score:.2%}", # Format as a percentage
#             "description": descriptive_diagnosis,
#             "recommendation": "For a definitive diagnosis and treatment plan, please consult with a local agricultural expert or extension service."
#         }

# vision_service = VisionService()


import base64
import io
from PIL import Image
from fastapi import UploadFile, Form
from shared.core.config import settings

class VisionService:
    def __init__(self):
        print("🌿 FieldScout AI: Initializing Unified Vision Service...")
        # Define expert prompts for different analysis modes
        self.prompts = {
            "disease_diagnosis": """
                You are an expert plant pathologist. Your task is to analyze this image for signs of disease or pest infestation.
                - **Diagnosis:** Concisely state the most likely health issue. If the plant appears healthy, state that.
                - **Confidence:** Rate your confidence in the diagnosis (High, Medium, or Low).
                - **Recommendation:** Provide a brief, actionable recommendation for a farmer.
                Structure your entire response in markdown format.
            """,
            "crop_identification": """
                You are an expert agronomist. Identify the crop in this image. 
                Provide the common name and, if possible, the species. 
                Mention any key visual characteristics you used for identification.
            """,
            "field_health_analysis": """
                You are an agricultural analyst reviewing an aerial or wide-angle photo of a field. 
                Analyze the overall health and condition. Look for patterns like discoloration (potential nutrient deficiency or irrigation issues), patchiness (potential pest infestation), or variations in growth. 
                Provide a summary of your observations and suggest areas that might require a closer, on-ground inspection.
            """
        }
        print("✅ Service is ready.")

    async def analyze_image(self, image_file: UploadFile, analysis_mode: str) -> dict:
        from litellm import acompletion
        
        print(f"🖼️ Analyzing image '{image_file.filename}' with mode: {analysis_mode}")
        image_bytes = await image_file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Select the prompt based on the user's chosen mode
        prompt = self.prompts.get(analysis_mode, "Describe this agricultural image.")

        try:
            response = await acompletion(
                model=f"ollama/{settings.VISION_MODEL}",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                api_base=settings.OLLAMA_API_BASE_URL
            )
            ai_response = response.choices[0].message.content
            return {"analysis": ai_response}
        except Exception as e:
            print(f"❌ ERROR in Vision Service: {e}")
            return {"analysis": f"Error analyzing image: {e}"}

vision_service = VisionService()