import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slackbot.bot import SlackBot
from config import PORT, APP_ENV, LOG_LEVEL


# --- Configure Logging ---
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),   
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Initialize Slack Bot ---
slack_app = SlackBot()

@asynccontextmanager
async def lifespan(app: FastAPI):
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

fastapi_app = FastAPI(
    title="TN Slackbot",
    lifespan=lifespan
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@fastapi_app.get("/")
async def root():
    return {"status": "ok", "message": "TN Slackbot is running"}

if __name__ == "__main__":
    uvicorn.run("main:fastapi_app", host="0.0.0.0", port=PORT, reload=(APP_ENV != "production"))
