from fastapi import APIRouter
from .agents.financial_agents import financial_agent
from pydantic import BaseModel, Field

router = APIRouter()

class PlanRequest(BaseModel):
    land_size: float = Field(..., gt=0, description="Size of land in acres")
    crop: str = Field(..., min_length=2, description="The crop to be planted")
    state: str = Field(..., min_length=2) # 👈 Add state
    district: str = Field(..., min_length=2) # 👈 Add district

@router.post("/generate-plan", tags=["Finance"])
async def generate_plan(request: PlanRequest):
    """Generates a high-level financial plan for a farmer."""
    plan = await financial_agent.get_financial_plan(
        request.land_size, request.crop, request.state, request.district
    )
    return plan