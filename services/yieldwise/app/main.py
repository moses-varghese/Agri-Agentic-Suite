from fastapi import FastAPI
from .api import router as finance_router
from fastapi.middleware.cors import CORSMiddleware
from shared.core.config import settings

app = FastAPI(
    title="YieldWise - Prototype 2",
    description="Financial planning and strategy for agriculture.",
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
    return {"status": "ok", "service": "YieldWise"}

app.include_router(finance_router)
print(f"🚀 YieldWise Service loaded in {settings.ENVIRONMENT} mode.")