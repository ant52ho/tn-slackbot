"""
Slack Bot Implementation

This module implements a Slack bot that handles various Slack events and integrates with a Jira ticket system.
The bot uses Slack's Socket Mode for real-time event handling and supports message events, reactions,
and private conversations for ticket creation.

Key Features:
- Real-time message handling
- Reaction-based ticket creation workflow
- Thread-based conversations
- Jira ticket integration
- Private message handling for sensitive operations

Environment Variables:
- SLACK_BOT_TOKEN: Bot User OAuth Token for Slack API
- SLACK_APP_TOKEN: App-Level Token for Socket Mode
- SLACK_THREAD_FETCH_LIMIT: Maximum number of messages to fetch from a thread
- LOG_LEVEL: Logging level configuration

Example:
    To initialize the bot:
    >>> bot = SlackBot()
    >>> await bot.start()
"""

import logging
from typing import Dict, Any, Optional
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from config import SLACK_BOT_TOKEN, SLACK_APP_TOKEN, SLACK_THREAD_FETCH_LIMIT, LOG_LEVEL
from fraud_chat_bot.superFraudBotV2 import JiraTicketBot

class SlackBot:
    """
    A Slack bot implementation that handles events and manages Jira ticket creation.
    
    This class manages the lifecycle of a Slack bot, handling various Slack events
    and coordinating with a Jira ticket bot for ticket creation workflows.
    
    Attributes:
        logger (logging.Logger): Logger instance for the bot
        slack_app (AsyncApp): Slack Bolt AsyncApp instance
        jira_ticket_bot (JiraTicketBot): Instance of the Jira ticket bot
        handler (AsyncSocketModeHandler): Socket mode handler for Slack events
    """
    
    def __init__(self) -> None:
        """
        Initialize the Slack bot with logging, Slack app, and event handlers.
        """
        # Logging is already configured in main.py, no need to configure again here
        self.logger: logging.Logger = logging.getLogger(__name__)
        
        # Initialize Slack app
        self.logger.info("Initializing Slack app...")
        self.slack_app = AsyncApp(token=SLACK_BOT_TOKEN)
        
        # Register event handlers
        self._register_event_handlers()

        # Initialize Jira ticket bot
        self.jira_ticket_bot = JiraTicketBot(self.slack_app)

    def _register_event_handlers(self) -> None:
        """
        Register event handlers for Slack events.
        
        Sets up handlers for:
        - Message events
        - Reaction added events
        """
        @self.slack_app.event("message")
        async def handle_message(event: Dict[str, Any], say: Any) -> None:
            """
            Handle incoming message events.
            
            Args:
                event (Dict[str, Any]): The Slack message event
                say (Any): Function to send messages back to Slack
            """
            self.logger.debug(f"Received message event: {event}")
            await self._handle_message_event(event, say)

        @self.slack_app.event("reaction_added")
        async def handle_reaction(event: Dict[str, Any], say: Any) -> None:
            """
            Handle reaction added events.
            
            Args:
                event (Dict[str, Any]): The Slack reaction event
                say (Any): Function to send messages back to Slack
            """
            self.logger.debug(f"Received reaction event: {event}")
            await self._handle_reaction_event(event)

    async def _handle_message_event(self, event: Dict[str, Any], say: Any) -> None:
        """
        Process incoming message events.
        
        Checks if the message is in a thread with a bot message and handles accordingly.
        
        Args:
            event (Dict[str, Any]): The Slack message event
            say (Any): Function to send messages back to Slack
        """
        self.logger.debug("=== Starting handle_message ===")
        if await self._top_thread_msg_is_bot(event):
            await self._handle_bot_chat_reply(event, say)
        self.logger.debug("=== Completed handle_message ===")

    async def _handle_reaction_event(self, event: Dict[str, Any]) -> None:
        """
        Process reaction added events.
        
        Currently handles the 'mega' reaction for ticket creation workflow.
        
        Args:
            event (Dict[str, Any]): The Slack reaction event
        """
        self.logger.debug("=== Starting handle_reaction ===")
        reaction = event.get("reaction")
        if reaction == "mega":
            await self._handle_mega_reaction(event)
        self.logger.debug("=== Completed handle_reaction ===")
    
    async def _top_thread_msg_is_bot(self, event: Dict[str, Any]) -> bool:
        """
        Check if the top message in a thread is from a bot.
        
        Args:
            event (Dict[str, Any]): The Slack event containing thread information
            
        Returns:
            bool: True if the top message is from a bot, False otherwise
        """
        if not event.get("thread_ts"):
            return False
        
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

    async def _top_thread_has_emoji(self, event: Dict[str, Any], emoji: str) -> bool:
        """
        Check if the top message in a thread has a specific emoji reaction.
        
        Args:
            event (Dict[str, Any]): The Slack event containing thread information
            emoji (str): The emoji to check for
            
        Returns:
            bool: True if the emoji is present, False otherwise
        """
        if not event.get("thread_ts"):
            return False
        
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
    
    async def _handle_bot_chat_reply(self, event: Dict[str, Any], say: Any) -> None:
        """
        Handle replies to bot messages in threads.
        
        Processes user replies to bot messages and forwards them to the Jira ticket bot.
        
        Args:
            event (Dict[str, Any]): The Slack message event
            say (Any): Function to send messages back to Slack
            
        Raises:
            Exception: If there's an error processing the message
        """
        try:
            self.logger.debug("Starting bot chat reply handler")
            if event.get("bot_id"):
                self.logger.info("Message is from a bot, skipping")
                return
            await self.jira_ticket_bot.handle_message(event)
            self.logger.debug("Completed bot chat reply handler")
            
        except Exception as e:
            self.logger.error(f"Error in handle_bot_chat_reply: {str(e)}", exc_info=True)
            await say("I encountered an error processing your message. Please try again.")

    async def _handle_mega_reaction(self, event: Dict[str, Any]) -> None:
        """
        Handle the mega reaction workflow for ticket creation.
        
        Initiates a private conversation with the user and sets up the Jira ticket bot
        for ticket creation based on the thread context.
        
        Args:
            event (Dict[str, Any]): The Slack reaction event
        """
        self.logger.debug("=== Starting handle_mega_reaction ===")
        self.logger.info(f"Event: {event}")
        reaction_item = event.get("item")
        reaction_channel = reaction_item.get("channel")
        reaction_ts = reaction_item.get("ts")
        
        # Get thread context
        res = await self.slack_app.client.conversations_replies(
            channel=reaction_channel,
            ts=reaction_ts,
            limit=SLACK_THREAD_FETCH_LIMIT
        )
        context = res.get("messages", [])
        msgs = [message.get("text") for message in context]
        msgs_str = "\n".join(msgs)

        # Initiate private conversation
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
        await self.jira_ticket_bot.init_chatbot_session(
            channel_id, 
            thread_ts, 
            original_channel_id=reaction_channel, 
            original_thread_ts=reaction_ts
        )
        self.logger.debug("=== Completed handle_mega_reaction ===")

    async def start(self) -> None:
        """
        Start the Slack bot and establish Socket Mode connection.
        
        Raises:
            Exception: If there's an error starting the bot
        """
        self.logger.info("Starting up Slack bot...")
        try:
            self.handler = AsyncSocketModeHandler(
                app=self.slack_app, 
                app_token=SLACK_APP_TOKEN
            )
            await self.handler.start_async()
            self.logger.info("Slack bot connected successfully")
        except Exception as e:
            self.logger.error(f"Failed to start Slack bot: {str(e)}", exc_info=True)
            raise

    async def stop(self) -> None:
        """
        Stop the Slack bot and clean up resources.
        
        Closes the Socket Mode handler and any Jira ticket bot resources.
        
        Raises:
            Exception: If there's an error during shutdown
        """
        self.logger.info("Shutting down Slack bot...")
        try:
            await self.handler.close_async()
            
            if hasattr(self, 'jira_ticket_bot'):
                await self.jira_ticket_bot.close()
            
            self.logger.info("Slack bot disconnected successfully")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}", exc_info=True) 
