"""
Jira Ticket Creation Bot

This module implements a sophisticated bot that creates Jira tickets based on Slack conversations.
It integrates with Slack's messaging system and uses OpenAI's GPT model to analyze conversations
and generate appropriate ticket details. The bot supports various workflows including ticket creation,
similarity search, and fraud notification handling.

Key Features:
- Automated Jira ticket creation from Slack conversations
- Similarity search using ChromaDB for finding related issues
- OpenAI-powered conversation analysis
- Multi-step ticket creation workflow
- Fraud notification handling
- Session management for ongoing conversations

Environment Variables:
- JIRA_SERVER: URL of the Jira server
- JIRA_USERNAME: Username for Jira authentication
- JIRA_PASSWORD: Password for Jira authentication
- JIRA_PROJECT: Project key for ticket creation
- REDIS_URL: URL for Redis session storage
- OPENAI_API_KEY: API key for OpenAI services
- REDIS_TTL_HOURS: Time-to-live for Redis sessions
- SLACK_THREAD_FETCH_LIMIT: Maximum messages to fetch from a thread
- LOG_LEVEL: Logging level configuration
- CHAT_DATA_PATH: Path to chat data for similarity search
- CHROMA_HOST: Host for ChromaDB
- CHROMA_PORT: Port for ChromaDB
- SIMILAR_ISSUES_COUNT: Number of similar issues to return

Example:
    To initialize the bot:
    >>> bot = JiraTicketBot(slack_app)
    >>> await bot.init_chatbot_session(channel, thread_ts, original_channel_id, original_thread_ts)
"""

import logging
import json
from langchain_openai import ChatOpenAI
from jira import JIRA
from typing import Any, Optional, Literal
from sessions.session_manager import SlackSessionManager, SlackSession
from slack_bolt import App
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, ConfigDict
from config import (
    JIRA_SERVER, JIRA_USERNAME, JIRA_PASSWORD, JIRA_PROJECT,
    REDIS_URL, OPENAI_API_KEY, REDIS_TTL_HOURS, SLACK_THREAD_FETCH_LIMIT,
    LOG_LEVEL, CHAT_DATA_PATH, CHROMA_HOST, CHROMA_PORT, SIMILAR_ISSUES_COUNT
)
from similarity_search.dataloader import ChatDataLoader
import chromadb
from chromadb.utils import embedding_functions

class TicketDetails(BaseModel):
    """
    Model for storing ticket creation details.
    
    This model defines the structure for ticket information extracted from conversations
    and used for Jira ticket creation.
    
    Attributes:
        title (str): Title of the Jira ticket
        description (str): Detailed description of the issue
        priority (str): Priority level from P0 to P4
        skip_details (bool): Flag to skip detailed ticket creation
        fraud_noc_pinged (bool): Flag to ping fraud_noc team
        follow_up_question (Optional[str]): Question to ask for more details
    """
    model_config = ConfigDict(extra="forbid")
    
    title: str = Field(description="title of the jira ticket")
    description: str = Field(description="description of the jira ticket")
    priority: str = Field(description="priority of the jira ticket, from P0 to P4")
    skip_details: bool = Field(description="User inputted flag to skip inputting details for this ticket")
    fraud_noc_pinged: bool = Field(description="User inputted flag to ping fraud_noc")
    follow_up_question: Optional[str] = Field(description="follow up question to ask the user to provide more details")

class ConfirmationDetails(BaseModel):
    """
    Model for storing ticket confirmation details.
    
    This model is used to store the final ticket details before creation
    and for user confirmation.
    
    Attributes:
        summary (str): Summary of the ticket
        description (str): Detailed description
        priority (str): Priority level
        fraud_noc_pinged (bool): Whether fraud_noc was pinged
    """
    model_config = ConfigDict(extra="forbid")
    
    summary: str
    description: str
    priority: str
    fraud_noc_pinged: bool

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the confirmation details to a dictionary.
        
        Returns:
            dict[str, Any]: Dictionary representation of the confirmation details
        """
        return {
            "summary": self.summary,
            "description": self.description,
            "priority": self.priority,
            "fraud_noc_pinged": self.fraud_noc_pinged
        }

logger: logging.Logger = logging.getLogger(__name__)

class JiraTicketBot:
    """
    A bot that creates Jira tickets from Slack conversations.
    
    This class manages the entire workflow of creating Jira tickets from Slack conversations,
    including conversation analysis, similarity search, and ticket creation.
    
    Attributes:
        session_manager (SlackSessionManager): Manages conversation sessions
        jira_client (JIRA): Jira API client
        slack_app (App): Slack Bolt app instance
        logger (logging.Logger): Logger instance
        chroma_client (Optional[chromadb.AsyncHttpClient]): ChromaDB client
        embedding_function (Optional[Any]): Embedding function for similarity search
        collection (Optional[Any]): ChromaDB collection
        cdl (ChatDataLoader): Data loader for similarity search
        llm (ChatOpenAI): OpenAI language model instance
    """
    
    def __init__(self, slack_app: App) -> None:
        """
        Initialize the Jira ticket bot.
        
        Args:
            slack_app (App): Slack Bolt app instance
        """
        self.session_manager = SlackSessionManager(REDIS_URL, REDIS_TTL_HOURS)
        self.jira_client = JIRA(server=JIRA_SERVER, basic_auth=(JIRA_USERNAME, JIRA_PASSWORD))
        self.slack_app = slack_app
        self.logger = logger
        
        # Initialize ChromaDB components
        self.chroma_client = None
        self.embedding_function = None
        self.collection = None

        # Note: Must be same as where you had loaded the data. 
        self.collection_name = "messages3"
        
        # Set up data loader for similarity search
        self.cdl = ChatDataLoader()

        # Initialize language model
        self.llm = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            api_key=OPENAI_API_KEY
        )
    
    async def init_chatbot_session(self, channel: str, thread_ts: str, original_channel_id: str, original_thread_ts: str) -> None:
        """
        Initialize a new chatbot session.
        Requires channel, thread_ts to post messages in.
        Requires original_channel_id, original_thread_ts to output result messages and read historical messages.
        
        Args:
            channel (str): Slack channel ID
            thread_ts (str): Thread timestamp
            original_channel_id (str): Original channel ID for reference
            original_thread_ts (str): Original thread timestamp for reference
        """
        # Get or create session
        session = await self.session_manager.get_session(channel, thread_ts)
        if not session:
            session = await self.session_manager.create_session(channel, thread_ts, original_channel_id, original_thread_ts)

        # Pass in dummy event 
        dummy = {
            "channel": channel,
            "thread_ts": thread_ts,
            "text": "",
        }

        await self._handle_event_for_current_step(session, dummy)
        return

    async def handle_message(self, event: dict[str, Any]) -> None:
        """
        Handle incoming Slack messages.
        
        Args:
            event (dict[str, Any]): Slack message event
        """
        logger.debug(f"Handling message {event.get('text')}")

        if not self._validate_event(event):
            logger.warning(f"Invalid event: {event}")
            return
        
        # Extract event details
        text = event.get('text')
        thread_ts = event.get('thread_ts')
        channel = event.get('channel')
        user = event.get('user')

        # Get or create session
        session = await self.session_manager.get_session(channel, thread_ts)
        if not session:
            session = await self.session_manager.create_session(channel, thread_ts)

        await self._handle_event_for_current_step(session, event)
        return
        
    def _validate_event(self, event: dict[str, Any]) -> bool:
        """
        Validate the incoming Slack event.
        
        Args:
            event (dict[str, Any]): Slack event to validate
            
        Returns:
            bool: True if event is valid, False otherwise
        """
        return event.get('text') and event.get('thread_ts') and event.get('channel') and event.get('user')

    async def _handle_event_for_current_step(self, session: SlackSession, event: dict[str, Any]) -> None:
        """
        Handle the event based on the current step in the workflow.
        Currently, a ticket moves from state to state: Initial -> Description -> Confirmation -> Finished
        However, depending on user input, a ticket may transition to confirmation or description state
        
        Args:
            session (SlackSession): Current conversation session
            event (dict[str, Any]): Slack event to handle
        """
        current_step = session.state.current_step

        if current_step == "initial":
            await self._handle_similarity_search_step(session, event)
        elif current_step == "description":
            await self._handle_description_step(session, event)
        elif current_step == "confirmation":
            await self._handle_confirmation_step(session, event)
        elif current_step == "finished":
            await self._handle_finished_step(session, event)
        else:
            self.logger.warning(f"Invalid session state: {session.state}")

        # Update session last activity
        await self.session_manager.update_session(session)
        return 

    async def _handle_similarity_search_step(self, session: SlackSession, event: dict[str, Any]) -> None:
        """
        Handle the similarity search step of the workflow.
        
        Searches for similar issues in the database and presents them to the user.
        
        Args:
            session (SlackSession): Current conversation session
            event (dict[str, Any]): Slack event to handle
        """
        if not self.chroma_client:
            self.chroma_client = await chromadb.AsyncHttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-ada-002"
            )
            self.collection = await self.chroma_client.get_or_create_collection(
                self.collection_name,
                embedding_function=self.embedding_function
            )
        
        # Format message for similarity search
        msg_df = await self.cdl.load_conversations(self.slack_app, session.original_channel_id, session.original_thread_ts)
        cleaned_msg_df = self.cdl.preprocess_data(msg_df)
        formatted_msg = cleaned_msg_df.iloc[0]['conversation']

        # Query similar issues
        results = await self.collection.query(
            query_texts=[formatted_msg],
            n_results=SIMILAR_ISSUES_COUNT
        )

        message = ""
        if results['metadatas']:
            message += "Here are some similar issues we have seen:\n"
            for metadata in results['metadatas'][0]:
                channel = metadata["channel_id"]
                thread_ts = str(metadata["thread_id"]).replace(".", "")
                link = f"https://textnow.slack.com/archives/{channel}/p{thread_ts}"
                message += f"{link}\n"

        # Move to description step
        session.state.current_step = "description"
        response = message if message else (
            "Hi! It looks like we couldn't find a similar issue. We will now create a ticket out of your message. "
            "At any point, please ping @fraud_noc to skip this process."
        )

        self.logger.debug(f"Sending response to channel {event.get('channel')} with thread_ts {event.get('thread_ts')}")
        res = await self.slack_app.client.chat_postMessage(
            channel=event.get('channel'),
            text=response,
            thread_ts=event.get('thread_ts')
        )
        self.logger.debug(f"Response sent: {res}")
        await self._handle_description_step(session, event)
        return

    async def _handle_description_step(self, session: SlackSession, event: dict[str, Any]) -> None:
        """
        Handle the description step of the workflow.
        
        Analyzes the conversation and generates ticket details using the language model.
        
        Args:
            session (SlackSession): Current conversation session
            event (dict[str, Any]): Slack event to handle
        """
        messages = await self._load_event_messages(event)

        # Create prompt template for ticket details
        prompt_template = """
            You are a fraud expert working on the Trust and Safety team for Texting Company A. 
            You must get enough information related to the user's issue to create a Jira ticket for another expert to later work on. 
            The expert has a good understanding of the system, so you only need to ask follow up questions on larger scope details. 
            
            Keep in mind you may only ask follow up questions through the json 'follow_up_question' field provided, and will otherwise
            not be asked to the user.
            You should aim to ask around 2-3 follow up questions. 
            Keep in mind all users work for Texting Company A, so you do not need to treat users with suspicion. 

            Do not leave follow_up_question blank if you do not have enough information. 
            Do not ping fraud noc if the user does not request for it. 

            You are given a Slack exchange between user(s) and yourself. Be sure to not confuse the user's message with your own.

            You should only answer relevant questions to Trust and Safety. Please remind the user in your follow up questions if 
            they are asking you to do something that is not related to Trust and Safety.

            Based on the context provided below, please extract and refine the key details needed 
            to create your Jira ticket. Determine if the issue is on an individual scope (e.g., 'I can't log in', 'I am getting rate limited') 
            or a broader fraud issue (e.g., a spike in registrations affecting a company metric).
            For individual issues: If data such as account details, device information, etc. is missing, include a follow-up question
            Otherwise, leave it blank.
            For broader issues: If details such as which dashboard you're using, or the specific date range of concern are missing, or etc., include a follow-up question.
            Otherwise, leave it blank.
            Use second-person language and only ask for details not already provided. Provide a concise title and a descriptive summary. 
            
            Format your response as JSON with these fields:
            - title (string): Summary of the issue.
            - description (string): Very detailed and refined problem description.
            - priority (string): Priority of the issue, from P0 to P4.
            - follow_up_question (string): Ask a follow up question to the user to provide more details if details of the description are missing. Leave blanki if none and are very confident in answer. 
            - skip_details (boolean): True if @fraud_noc mentioned, False otherwise (default: False).
            - fraud_noc_pinged (boolean): True if @fraud_noc mentioned, False otherwise (default: False).
            Use second-person language and only request information not already provided.
            
            Context:
            {messages}

            Format instructions
            {format_instructions}
        """

        parser = PydanticOutputParser(pydantic_object=TicketDetails)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["messages"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # Generate ticket details using language model
        chain = prompt | self.llm | parser
        result: TicketDetails = chain.invoke({"messages": messages})
        self.logger.info(
            f"ChatGPT result: {json.dumps(result.model_dump(), indent=2)}"
        )

        # Handle different outcomes based on the result
        if result.skip_details:
            await self._handle_skip_details(session, event, result)
        elif result.fraud_noc_pinged:
            await self._handle_ping_fraud_noc(session, event, result)
        elif result.follow_up_question:
            await self._handle_send_follow_up_question(session, event, result)
        else:
            await self._handle_send_ticket_confirmation(session, event, result)
        
        return

    async def _load_event_messages(self, event: dict[str, Any]) -> str:
        """
        Load and format messages from a Slack thread using its channel and thread_ts.
        
        Args:
            event (dict[str, Any]): Slack event containing thread information
            
        Returns:
            str: Formatted string of messages in the thread
        """
        if not event.get("thread_ts"):
            self.logger.warning(f"Failed to load event messages: no thread_ts found in event: {event}")
            return ""
        
        # Get all messages in the thread
        response = await self.slack_app.client.conversations_replies(
            channel=event["channel"],
            ts=event["thread_ts"],
            limit=SLACK_THREAD_FETCH_LIMIT
        )
        messages = response.get("messages", [])

        # Format messages with user numbers
        encountered_users = {}
        formatted_messages = []
        for msg in messages:
            user = msg.get("user", "Unknown")
            text = msg.get("text", "")

            if user in encountered_users:
                formatted_messages.append(f"User{encountered_users[user]}: {text}")
            else:
                userNum = len(encountered_users) + 1
                formatted_messages.append(f"User{userNum}: {text}")
                encountered_users[user] = userNum

        return "\n".join(formatted_messages)
    
    async def _send_confirmation_message(self, event: dict[str, Any], session: SlackSession, confirmation_details: ConfirmationDetails) -> None:
        """
        Send a confirmation message to the user with ticket details.
        
        Args:
            event (dict[str, Any]): Slack event
            session (SlackSession): Current conversation session
            confirmation_details (ConfirmationDetails): Details to confirm
        """
        preview_message = (
            "Here is a preview of the ticket you will create.\n"
            f"- Title: {confirmation_details.summary}\n"
            f"- Description: {confirmation_details.description}\n"
            f"- Priority: {confirmation_details.priority}"
        )

        if confirmation_details.priority in ["P0", "P1", "P2"]:
            preview_message += f"\n\nNOTE: Priority {confirmation_details.priority} detected. Will autoping fraud_noc."
        elif confirmation_details.fraud_noc_pinged:
            preview_message += "\n\nWill ping fraud_noc for review."

        preview_message += "\n\nIf this is correct, please reply strictly with 'confirm' to create the ticket. Otherwise, reply with the details you would like to change."

        await self.slack_app.client.chat_postMessage(
            channel=event.get('channel'),
            text=preview_message,
            thread_ts=event.get('thread_ts')
        )

        # Update session state
        session.state.ticket_data.summary = confirmation_details.summary
        session.state.ticket_data.description = confirmation_details.description
        session.state.ticket_data.priority = confirmation_details.priority
        session.state.ticket_data.fraud_noc_pinged = confirmation_details.fraud_noc_pinged
        return 
    
    async def _create_jira_ticket(self, session: SlackSession, summary: str, description: str, priority: str) -> str:
        """
        Create a Jira ticket with the provided details.
        
        Args:
            session (SlackSession): Current conversation session
            summary (str): Ticket summary
            description (str): Ticket description
            priority (str): Ticket priority
            
        Returns:
            str: Success message with ticket key
        """
        new_issue = self.jira_client.create_issue(
            project=JIRA_PROJECT,
            summary=summary,
            description=description,
            issuetype={"name": "Task"},
            priority={"name": priority}
        )
        session.state.ticket_data.ticket_key = new_issue.key
        message = (f"✅ Ticket `{new_issue.key}` created successfully! "
                   f"View the ticket: https://textnow.atlassian.net/browse/{new_issue.key}")
        return message

    async def _handle_skip_details(self, session: SlackSession, event: dict[str, Any], chat_result: TicketDetails) -> None:
        """
        Handle the skip details workflow.
        
        Args:
            session (SlackSession): Current conversation session
            event (dict[str, Any]): Slack event
            chat_result (TicketDetails): Chat analysis result
        """
        confirmation_details = ConfirmationDetails(
            summary="Fraud Squad - Low Context Please See Channel",
            description=f"Slack Thread: {event.get('thread_ts')}",
            priority=chat_result.priority,
            fraud_noc_pinged=chat_result.fraud_noc_pinged
        )
        await self._send_confirmation_message(event, session, confirmation_details)
        session.state.current_step = "confirmation"
        return
    
    async def _handle_ping_fraud_noc(self, session: SlackSession, event: dict[str, Any], chat_result: TicketDetails) -> None:
        """
        Handle the fraud_noc ping workflow.
        
        Args:
            session (SlackSession): Current conversation session
            event (dict[str, Any]): Slack event
            chat_result (TicketDetails): Chat analysis result
        """
        confirmation_details = ConfirmationDetails(
            summary=chat_result.title,
            description=chat_result.description,
            priority=chat_result.priority,
            fraud_noc_pinged=True
        )
        await self._send_confirmation_message(event, session, confirmation_details)
        session.state.current_step = "confirmation"
        session.state.ticket_data.fraud_noc_pinged = True
        return
    
    async def _handle_send_follow_up_question(self, session: SlackSession, event: dict[str, Any], chat_result: TicketDetails) -> None:
        """
        Handle sending a follow-up question to the user.
        
        Args:
            session (SlackSession): Current conversation session
            event (dict[str, Any]): Slack event
            chat_result (TicketDetails): Chat analysis result
        """
        message = (
            chat_result.follow_up_question + "\n\n"
            "At any point, please let us know if you would like to ping fraud_noc and skip this process."
        )

        await self.slack_app.client.chat_postMessage(
            channel=event.get('channel'),
            text=message,
            thread_ts=event.get('thread_ts')
        )
        return
    
    async def _handle_send_ticket_confirmation(self, session: SlackSession, event: dict[str, Any], chat_result: TicketDetails) -> None:
        """
        Handle sending ticket confirmation to the user.
        
        Args:
            session (SlackSession): Current conversation session
            event (dict[str, Any]): Slack event
            chat_result (TicketDetails): Chat analysis result
        """
        confirmation_details = ConfirmationDetails(
            summary=chat_result.title,
            description=chat_result.description,
            priority=chat_result.priority,
            fraud_noc_pinged=chat_result.fraud_noc_pinged
        )
        await self._send_confirmation_message(event, session, confirmation_details)
        session.state.current_step = "confirmation"
        return
    
    async def _handle_confirmation_step(self, session: SlackSession, event: dict[str, Any]) -> None:
        """
        Handle the confirmation step of the workflow.
        
        Args:
            session (SlackSession): Current conversation session
            event (dict[str, Any]): Slack event
        """
        if event.get("text") == "confirm":
            create_msg = await self._create_jira_ticket(
                session, 
                session.state.ticket_data.summary, 
                session.state.ticket_data.description, 
                session.state.ticket_data.priority
            )
            if session.state.ticket_data.fraud_noc_pinged:
                create_msg += "\n\n @fraud_noc was pinged."

            # Post in private DM to user
            await self.slack_app.client.chat_postMessage(
                channel=event.get('channel'),
                text=create_msg,
                thread_ts=event.get('thread_ts')
            )

            # Post in original session channel
            await self.slack_app.client.chat_postMessage(
                channel=session.original_channel_id,
                text=create_msg,
                thread_ts=session.original_thread_ts
            )

            session.state.current_step = "finished"
        else:
            session.state.current_step = "description"
            await self._handle_description_step(session, event)
        
        return
    
    async def _handle_finished_step(self, session: SlackSession, event: dict[str, Any]) -> None:
        """
        Handle the finished step of the workflow.
        
        Args:
            session (SlackSession): Current conversation session
            event (dict[str, Any]): Slack event
        """
        await self.slack_app.client.chat_postMessage(
            channel=event.get('channel'),
            text=f"Chatbot has finished processing this thread. To continue chatting, please react/re-react to another message",
            thread_ts=event.get('thread_ts')
        )
        return
        
    async def close(self) -> None:
        """
        Close all resources used by the bot.
        
        Closes the Redis connection and any other resources.
        """
        try:
            await self.session_manager.close()
            self.logger.info("JiraTicketBot resources closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing JiraTicketBot resources: {str(e)}")
