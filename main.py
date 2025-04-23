"""
TN Slackbot

This module serves as the main entrypoint for the TN Slackbot application. It initializes and runs a FastAPI server
that manages the Slack bot's lifecycle using Socket Mode. The application handles both HTTP requests and maintains
a persistent WebSocket connection to Slack.

Key Features:
- FastAPI server with CORS support
- Slack bot integration via Socket Mode
- Graceful startup and shutdown handling
- Configurable logging
- Environment-based configuration

Environment Variables:
- SERVER_PORT: Port number for the FastAPI server
- APP_ENV: Application environment (e.g., "development", "production")
- LOG_LEVEL: Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR")

Example:
    To run the application:
    $ python main.py
"""

import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slackbot.bot import SlackBot
from config import SERVER_PORT, APP_ENV, LOG_LEVEL
from typing import AsyncGenerator

# --- Configure Logging ---
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),   
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Initialize Slack Bot ---
slack_app = SlackBot()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan manager that handles the Slack bot's lifecycle.
    
    This function manages the startup and shutdown of the Slack bot's Socket Mode connection.
    It ensures proper initialization and cleanup of resources.
    
    Args:
        app (FastAPI): The FastAPI application instance
        
    Yields:
        None: Control is yielded to FastAPI while the application is running
    """
    # Startup
    logger.info("⚡ Starting Slack SocketModeHandler...")
    slack_task = asyncio.create_task(slack_app.start())

    yield  # FastAPI is live

    # Shutdown
    logger.info("🛑 Shutting down Slack SocketModeHandler...")
    await slack_app.stop()
    slack_task.cancel()
    try:
        await slack_task
    except asyncio.CancelledError:
        logger.info("Slack bot task cancelled cleanly.")

# Initialize FastAPI application with lifespan management
fastapi_app = FastAPI(
    title="TN Slackbot",
    description="A Slack bot for TN with FastAPI backend",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@fastapi_app.get("/")
async def root() -> dict[str, str]:
    """
    Root endpoint that returns the application status.
    
    Returns:
        dict: A dictionary containing the application status and message
    """
    return {"status": "ok", "message": "TN Slackbot is running"}

if __name__ == "__main__":
    # Run the FastAPI application with uvicorn
    # Enable auto-reload in non-production environments
    uvicorn.run(
        "main:fastapi_app",
        host="0.0.0.0",
        port=SERVER_PORT,
        reload=(APP_ENV != "production")
    )
