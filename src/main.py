from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from src.config import SLACK_BOT_TOKEN, SLACK_APP_TOKEN
from src.handlers.handleReactions import handle_david_reaction, handle_jira_creation_chat, handle_fraud_noc_ping
import logging
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="TN Slackbot")

# Initialize Slack app
logger.info("Initializing Slack app...")
slack_app = App(token=SLACK_BOT_TOKEN)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "TN Slackbot is running"}

# todo: add event handler for message
# @slack_app.event("message")

@slack_app.event("reaction_added")
def handle_reaction(event, say):
    """Handle reaction_added events"""
    logger.info("=== Starting handle_reaction ===")
    try:
        logger.info(f"Received event: {event}")
        # Extract event data
        reaction = event.get("reaction")
        user = event.get("user")
        item = event.get("item", {})
        channel = item.get("channel")
        ts = item.get("ts")

        # get extra information
        logger.info("Fetching conversation messages...")
        messages = slack_app.client.conversations_history(
            channel=channel,
            latest=ts,
            limit=1,
            inclusive=True
        )["messages"]
        original_message = messages[0]
        original_sender_id = original_message.get("user")

        logger.info(f"Original sender: {original_sender_id}")
        
        # Handle david reaction
        if reaction == "david":
            handle_david_reaction(slack_app, event, say)
        elif reaction == "thumbsup" and original_sender_id == "U08JWRHKQUW":
            handle_jira_creation_chat(slack_app, event, say)
        # elif reaction == "trust-and-safety" and original_sender_id == "U08JWRHKQUW":
        #     handle_fraud_noc_ping(slack_app, event, say)
        
    except Exception as e:
        logger.error(f"Error in handle_reaction: {str(e)}", exc_info=True)
    finally:
        logger.info("=== Completed handle_reaction ===")

@app.on_event("startup")
async def startup_event():
    """
    Initialize Socket Mode handler on startup
    """
    logger.info("Starting up Slack bot...")
    try:
        handler = SocketModeHandler(app=slack_app, app_token=SLACK_APP_TOKEN)
        # Start the handler in a separate thread
        thread = threading.Thread(target=handler.start, daemon=True)
        thread.start()
        logger.info("Slack bot connected successfully")
    except Exception as e:
        logger.error(f"Failed to start Slack bot: {str(e)}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup Socket Mode handler on shutdown
    """
    logger.info("Shutting down Slack bot...")
    # Bolt handles cleanup automatically
    logger.info("Slack bot disconnected successfully") 