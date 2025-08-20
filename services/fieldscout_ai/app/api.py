from fastapi import APIRouter, UploadFile, File, Form
from .services.vision_service import vision_service

router = APIRouter()

# @router.post("/diagnose", tags=["Vision"])
# async def diagnose_crop_image(image: UploadFile = File(...)):
#     """Accepts an image and returns a diagnosis."""
#     result = await vision_service.analyze_image(image)
#     return result


@router.post("/diagnose", tags=["Vision"])
async def diagnose_crop_image(
    image: UploadFile = File(...),
    analysis_mode: str = Form("disease_diagnosis") # Receive the mode from the form
):
    result = await vision_service.analyze_image(image, analysis_mode)
    return result