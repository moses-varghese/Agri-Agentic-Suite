from transformers import AutoImageProcessor, BeitForImageClassification

model_name = "microsoft/beit-base-patch16-224-pt22k-ft22k"

print(f"Downloading Hugging Face model: {model_name}")
# This will download and cache the model and processor
AutoImageProcessor.from_pretrained(model_name)
BeitForImageClassification.from_pretrained(model_name)
print("Download complete.")