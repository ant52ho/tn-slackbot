'''
superFraudBotV2 integrates with the slackbot without
opening a separate streamlit app. 

It is designed to interact directly with the slackbot by 
handling the message events and updating the session state.
'''

import logging
import json
from langchain_openai import ChatOpenAI
from jira import JIRA
from typing import Any
from sessions.session_manager import SlackSessionManager, SlackSession
from slack_bolt import App
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal
from config import JIRA_SERVER, JIRA_USERNAME, JIRA_PASSWORD, JIRA_PROJECT, REDIS_URL, OPENAI_API_KEY, REDIS_TTL_HOURS, SLACK_THREAD_FETCH_LIMIT, LOG_LEVEL

class TicketDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    # ticket fields
    title: str = Field(description="title of the jira ticket")
    description: str = Field(description="description of the jira ticket")
    priority: str = Field(description="priority of the jira ticket, from P0 to P4")

    # ticket creation flags
    skip_details: bool = Field(description="User inputted flag to skip inputting details for this ticket")
    fraud_noc_pinged: bool = Field(description="User inputted flag to ping fraud_noc")
    
    # follow up question for model to ask
    follow_up_question: Optional[str] = Field(description="follow up question to ask the user to provide more details")



class ConfirmationDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    summary: str
    description: str
    priority: str
    fraud_noc_pinged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "description": self.description,
            "priority": self.priority,
            "fraud_noc_pinged": self.fraud_noc_pinged
        }
    
# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger: logging.Logger = logging.getLogger(__name__)

# The general parameters for the bot
class JiraTicketBot:
    def __init__(self, slack_app: App):
        self.session_manager = SlackSessionManager(REDIS_URL, REDIS_TTL_HOURS)
        self.jira_client = JIRA(server=JIRA_SERVER, basic_auth=(JIRA_USERNAME, JIRA_PASSWORD))
        self.slack_app = slack_app
        self.logger = logger
    
        # set up llm
        self.llm = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            api_key=OPENAI_API_KEY
        )
    
    async def init_chatbot_session(self, channel: str, thread_ts: str, original_channel_id: str, original_thread_ts: str) -> None:
        """
        Start the workflow.
        """
        # Get or create session
        session = await self.session_manager.get_session(channel, thread_ts)
        if not session:
            session = await self.session_manager.create_session(channel, thread_ts, original_channel_id, original_thread_ts)

        dummy = {
            "channel": channel,
            "thread_ts": thread_ts,
            "text": "",
        }

        # needs to handle message & session management
        await self._handle_event_for_current_step(session, dummy)
        return
        

    async def handle_message(self, event: dict[str, Any]) -> None:
        """
        Handle a message event from Slack.
        """
        logger.debug(f"Handling message {event.get('text')}")

        # validate event 
        if not self._validate_event(event):
            logger.warning(f"Invalid event: {event}")
            return
        
        # unpack event
        text = event.get('text')
        thread_ts = event.get('thread_ts')
        channel = event.get('channel')
        user = event.get('user')

        # Get or create session
        session = await self.session_manager.get_session(channel, thread_ts)
        if not session:
            session = await self.session_manager.create_session(channel, thread_ts)

        # needs to handle message & session management
        await self._handle_event_for_current_step(session, event)
        return
        
    def _validate_event(self, event: dict[str, Any]) -> bool:
        """
        Validate the event.
        """
        return event.get('text') and event.get('thread_ts') and event.get('channel') and event.get('user')

    async def _handle_event_for_current_step(self, session: SlackSession, event: dict[str, Any]) -> None:
        """
        Handle the event for the current step.
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

        # update session last activity
        # python passes objects by reference, so session object is updated
        await self.session_manager.update_session(session)
        return 


    async def _handle_similarity_search_step(self, session: SlackSession, event: dict[str, Any]) -> None:
        """
        Handle the similarity search step.
        """
        # this is not implemented yet, so we just move to the description step
        session.state.current_step = "description"
        response = (
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
        Handle the initial step.
        """
        messages = await self._load_event_messages(event)

        # fill out jira ticket responses based on description
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

        chain = prompt | self.llm | parser
        result: TicketDetails = chain.invoke({"messages": messages})
        self.logger.info(
            f"ChatGPT result: {json.dumps(result.model_dump(), indent=2)}"
        )

        if result.skip_details:
            # early exit with empty ticket and message to check slack
            await self._handle_skip_details(session, event, result)
            return
        elif result.fraud_noc_pinged:
            # early exit with ticket and message to fraud_noc
            await self._handle_ping_fraud_noc(session, event, result)
            return
        elif result.follow_up_question:
            # early exit with follow up question
            await self._handle_send_follow_up_question(session, event, result)
        else:
            # send confirmation of ticket with current result
            await self._handle_send_ticket_confirmation(session, event, result)
        
        return

    async def _load_event_messages(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Load the messages for the event.
        """
        if not event.get("thread_ts"):
            return False
        
        # get all messages in the thread
        response = await self.slack_app.client.conversations_replies(
            channel=event["channel"],
            ts=event["thread_ts"],
            limit=SLACK_THREAD_FETCH_LIMIT
        )
        messages = response.get("messages", [])

        # format messages
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

        messages_str = "\n".join(formatted_messages)
        return messages_str
    
    async def _send_confirmation_message(self, event: dict[str, Any], session: SlackSession, confirmation_details: ConfirmationDetails) -> None:
        """
        Send a confirmation message to the user and loads data into session to send
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

        # Update ticket_data fields
        session.state.ticket_data.summary = confirmation_details.summary
        session.state.ticket_data.description = confirmation_details.description
        session.state.ticket_data.priority = confirmation_details.priority
        session.state.ticket_data.fraud_noc_pinged = confirmation_details.fraud_noc_pinged
        return 
    
    async def _create_jira_ticket(self, session: SlackSession, summary: str, description: str, priority: str) -> str:
        """
        Create a Jira ticket.
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
        Handle the skip details step.
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
        Handle the send follow up question step.
        """
        message = (
            chat_result.follow_up_question + "\n\n"
            "At any point, please let us know if you would like to ping fraud_noc and skip this process."
        )

        res = await self.slack_app.client.chat_postMessage(
            channel=event.get('channel'),
            text=message,
            thread_ts=event.get('thread_ts')
        )
        return
    
    async def _handle_send_ticket_confirmation(self, session: SlackSession, event: dict[str, Any], chat_result: TicketDetails) -> None:
        """
        Handle the send ticket confirmation step.
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
        Handle the confirmation step.
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

            # post in private dm to user
            await self.slack_app.client.chat_postMessage(
                channel=event.get('channel'),
                text=create_msg,
                thread_ts=event.get('thread_ts')
            )

            # post in original session channel
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
        Handle the finished step.
        """
        await self.slack_app.client.chat_postMessage(
            channel=event.get('channel'),
            text=f"Chatbot has finished processing this thread. To continue chatting, please react/re-react to another message",
            thread_ts=event.get('thread_ts')
        )
        return
        
    async def close(self) -> None:
        """Close all resources used by the bot."""
        try:
            # Close Redis connection
            await self.session_manager.close()
            self.logger.info("JiraTicketBot resources closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing JiraTicketBot resources: {str(e)}")
