"""
run backend
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.api.routers import clip_router

app = FastAPI(
    title="Hermes Backend",
    description="This is backend API endpoint for Hermes",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(clip_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
