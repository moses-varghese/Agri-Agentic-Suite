# from fastapi import FastAPI
# from fastapi.responses import PlainTextResponse
# from .api import router as sms_router
# from shared.core.config import settings

# app = FastAPI(
#     title="GroundTruth AI - Prototype 1",
#     description="Accessible agricultural advice via SMS.",
#     version="0.1.0"
# )

# @app.get("/", tags=["Health Check"])
# def read_root():
#     """A simple health check endpoint."""
#     return {"status": "ok", "service": "GroundTruth AI"}

# # Override the default response class for the Twilio endpoint
# # This ensures the Content-Type header is 'text/xml' as required by Twilio.
# app.include_router(sms_router, default_response_class=PlainTextResponse)

# print(f"🚀 GroundTruth AI Service loaded in {settings.ENVIRONMENT} mode.")


from fastapi import FastAPI
from .api import router as query_router
from shared.core.config import settings
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="GroundTruth AI - Prototype 1",
    description="Accessible agricultural advice via local RAG.",
    version="0.2.0"
)

#(Note: For production, you would restrict allow_origins to your specific frontend domain, but "*" is fine for development.)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "service": "GroundTruth AI"}

app.include_router(query_router)

print(f"🚀 GroundTruth AI Service loaded in {settings.ENVIRONMENT} mode.")