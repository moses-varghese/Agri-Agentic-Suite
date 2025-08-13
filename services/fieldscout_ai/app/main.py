from fastapi import FastAPI
from .api import router as vision_router
from fastapi.middleware.cors import CORSMiddleware
from shared.core.config import settings

app = FastAPI(
    title="FieldScout AI - Prototype 3",
    description="On-device crop disease diagnosis.",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "service": "FieldScout AI"}

app.include_router(vision_router)
print(f"🚀 FieldScout AI Service loaded in {settings.ENVIRONMENT} mode.")