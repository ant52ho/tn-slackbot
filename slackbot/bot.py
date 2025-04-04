import logging
from typing import Dict, Any
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from config import SLACK_BOT_TOKEN, SLACK_APP_TOKEN, SLACK_THREAD_FETCH_LIMIT, LOG_LEVEL
from fraud_chat_bot.src.superFraudBotV2 import JiraTicketBot

class SlackBot:
    def __init__(self):
        # Set up logging
        self._setup_logging()
        
        # Initialize Slack app
        self.logger.info("Initializing Slack app...")
        self.slack_app = AsyncApp(token=SLACK_BOT_TOKEN)
        
        # Register event handlers
        self._register_event_handlers()

        # set up jira ticket bot
        self.jira_ticket_bot = JiraTicketBot(self.slack_app)

    def _setup_logging(self):
        """Initialize logging configuration"""
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL, logging.DEBUG),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger: logging.Logger = logging.getLogger(__name__)

    def _register_event_handlers(self):
        """Register Slack event handlers"""
        @self.slack_app.event("message")
        async def handle_message(event, say):
            self.logger.debug(f"Received message event: {event}")
            await self._handle_message_event(event, say)

        @self.slack_app.event("reaction_added")
        async def handle_reaction(event, say):
            self.logger.debug(f"Received reaction event: {event}")
            await self._handle_reaction_event(event)

    async def _handle_message_event(self, event, say):
        """Handle message events"""
        self.logger.debug("=== Starting handle_message ===")
        if await self._top_thread_msg_is_bot(event):
            await self._handle_bot_chat_reply(event, say)
        self.logger.debug("=== Completed handle_message ===")

    async def _handle_reaction_event(self, event):
        """Handle reaction_added events"""
        self.logger.debug("=== Starting handle_reaction ===")
        reaction = event.get("reaction")
        if reaction == "mega":
            await self._handle_mega_reaction(event)

        self.logger.debug("=== Completed handle_reaction ===")
    
    async def _top_thread_msg_is_bot(self, event):
        """Check if the top thread message is from a bot"""
        # no thread_ts means convo is not thread
        if not event.get("thread_ts"):
            return False
        
        # get top thread from event thread_ts
        result = await self.slack_app.client.conversations_replies(
            channel=event["channel"],
            ts=event["thread_ts"],
            limit=1
        )

        thread_messages = result.get("messages", [])
        if len(thread_messages) > 0:
            original_message = thread_messages[0]
            if original_message.get("bot_id"):
                self.logger.debug(f"Top thread message is from a bot")
                return True
            
        self.logger.debug(f"Top thread message is not from a bot")
        return False

    async def _top_thread_has_emoji(self, event, emoji):
        """Check if the top thread has an emoji reaction"""
        # no thread_ts means convo is not thread
        if not event.get("thread_ts"):
            return False
        
        # get top thread from event thread_ts
        result = await self.slack_app.client.conversations_replies(
            channel=event["channel"],
            ts=event["thread_ts"],
            limit=1
        )

        thread_messages = result.get("messages", [])
        if len(thread_messages) > 0:
            original_message = thread_messages[0]
            reactions = original_message.get("reactions", [])
            for reaction in reactions:
                if reaction.get("name") == emoji:
                    self.logger.debug(f"Top thread has emoji: {emoji}")
                    return True
            
        self.logger.debug(f"Top thread does not have emoji: {emoji}")
        return False
    
    async def _handle_bot_chat_reply(self, event: Dict[str, Any], say) -> None:
        """Handle chat reply events"""
        try:
            self.logger.debug("Starting bot chat reply handler")
            # Return early if the message is from a bot
            if event.get("bot_id"):
                self.logger.info("Message is from a bot, skipping")
                return
            await self.jira_ticket_bot.handle_message(event)
            
            self.logger.debug("Completed bot chat reply handler")
            
        except Exception as e:
            self.logger.error(f"Error in handle_bot_chat_reply: {str(e)}", exc_info=True)
            await say("I encountered an error processing your message. Please try again.")


    async def _handle_mega_reaction(self, event: dict) -> None:
        """Handle the mega reaction workflow."""
        self.logger.debug("=== Starting handle_mega_reaction ===")
        self.logger.info(f"Event: {event}")
        reaction_item = event.get("item")
        reaction_channel = reaction_item.get("channel")
        reaction_ts = reaction_item.get("ts")
        
        # get the context of the message
        res = await self.slack_app.client.conversations_replies(
            channel=reaction_channel,
            ts=reaction_ts,
            limit=SLACK_THREAD_FETCH_LIMIT
        )
        context = res.get("messages", [])
        msgs = [message.get("text") for message in context]
        msgs_str = "\n".join(msgs)

        # initiate a private conversation with the user
        user_id = event.get("user")
        response = await self.slack_app.client.conversations_open(users=user_id)
        channel_id = response["channel"]["id"]

        message = (
            "================================================"
            "\n\nHello! I'll help you create a ticket for Trust & Safety. "
            f"\n\nContext:\n\n{msgs_str}"
            "\n\nPlease reply in this thread to start the conversation."
            "\n\n================================================"
        )
        response = await self.slack_app.client.chat_postMessage(
            channel=channel_id, 
            text=message
        )
        thread_ts = response["ts"]
        await self.jira_ticket_bot.init_chatbot_session(channel_id, thread_ts, original_channel_id=reaction_channel, original_thread_ts=reaction_ts)
        self.logger.debug("=== Completed handle_mega_reaction ===")

    async def start(self):
        """Start the Slack bot"""
        self.logger.info("Starting up Slack bot...")
        try:
            self.handler = AsyncSocketModeHandler(app=self.slack_app, app_token=SLACK_APP_TOKEN)
            await self.handler.start_async()
            self.logger.info("Slack bot connected successfully")
        except Exception as e:
            self.logger.error(f"Failed to start Slack bot: {str(e)}", exc_info=True)
            raise

    async def stop(self):
        """Stop the Slack bot"""
        self.logger.info("Shutting down Slack bot...")
        try:
            # close slack socket mode handler
            await self.handler.close_async()

            # Close JiraTicketBot resources
            if hasattr(self, 'jira_ticket_bot'):
                await self.jira_ticket_bot.close()
            
            # Close any other resources here
            self.logger.info("Slack bot disconnected successfully")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}", exc_info=True) 