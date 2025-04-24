import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from config import SLACK_BOT_TOKEN, SLACK_APP_TOKEN

# Slack setup
slack_app = AsyncApp(token=SLACK_BOT_TOKEN)
socket_handler = AsyncSocketModeHandler(slack_app, app_token=SLACK_APP_TOKEN)

@slack_app.event("reaction_added")
async def handle_mention(event, say):
    await say(f"Hi <@{event['user']}> 👋 (I'm aliveee!)")

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("⚡ Starting Slack SocketModeHandler...")
    slack_task = asyncio.create_task(socket_handler.start_async())
    yield
    # Shutdown
    print("🛑 Shutting down Slack SocketModeHandler...")
    await socket_handler.close_async()
    slack_task.cancel()
    try:
        await slack_task
    except asyncio.CancelledError:
        pass

# FastAPI app with lifespan context
fastapi_app = FastAPI(lifespan=lifespan)

@fastapi_app.get("/")
async def root():
    return {"message": "FastAPI + Slack Bolt is live"}