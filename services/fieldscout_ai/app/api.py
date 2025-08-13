from fastapi import APIRouter, UploadFile, File
from .services.vision_service import vision_service

router = APIRouter()

@router.post("/diagnose", tags=["Vision"])
async def diagnose_crop_image(image: UploadFile = File(...)):
    """Accepts an image and returns a diagnosis."""
    result = await vision_service.analyze_image(image)
    return result