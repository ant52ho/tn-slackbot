"""
Session Management Module

This module provides session management functionality for the Slack bot using Redis as a storage backend.
It handles the persistence and retrieval of conversation states, ticket data, and session information.

Key Features:
- Redis-based session storage
- Automatic session expiration
- Ticket metadata management
- Session state tracking
- Error handling and logging

Example:
    To use the session manager:
    >>> manager = SlackSessionManager(redis_url, redis_ttl_hours)
    >>> session = await manager.create_session(channel_id, thread_ts, original_channel_id, original_thread_ts)
"""

import logging
from datetime import timedelta
import redis.asyncio as redis
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class TicketData(BaseModel):
    """
    Model for storing ticket-related information.
    
    This model represents the data structure for a Jira ticket being created
    through the Slack bot conversation.
    
    Attributes:
        summary (str): Ticket summary/title
        description (str): Detailed ticket description
        priority (str): Ticket priority level (P0-P4)
        fraud_noc_pinged (bool): Whether fraud_noc was notified
        ticket_key (str): Jira ticket key after creation
    """
    summary: str = ""
    description: str = ""
    priority: str = "P4"
    fraud_noc_pinged: bool = False
    ticket_key: str = ""

class SessionState(BaseModel):
    """
    Model for storing the current state of a conversation session.
    
    This model tracks the progress of a ticket creation workflow
    and stores associated ticket data.
    
    Attributes:
        current_step (str): Current step in the workflow
        ticket_data (TicketData): Associated ticket information
    """
    current_step: str = "initial"
    ticket_data: TicketData = Field(default_factory=TicketData)

class SlackSession(BaseModel):
    """
    Model for storing a Slack conversation session.
    
    This model represents a complete session for a Slack conversation,
    including thread information and session state.
    
    Attributes:
        thread_ts (str): Thread timestamp
        channel_id (str): Channel ID
        original_channel_id (str): Original channel ID for reference
        original_thread_ts (str): Original thread timestamp for reference
        state (SessionState): Current session state
    """
    thread_ts: str
    channel_id: str
    original_channel_id: str
    original_thread_ts: str
    state: SessionState = Field(default_factory=SessionState)

    @classmethod
    def create(cls, thread_ts: str, channel_id: str, original_channel_id: str, original_thread_ts: str) -> 'SlackSession':
        """
        Create a new Slack session instance.
        
        Args:
            thread_ts (str): Thread timestamp
            channel_id (str): Channel ID
            original_channel_id (str): Original channel ID
            original_thread_ts (str): Original thread timestamp
            
        Returns:
            SlackSession: New session instance
        """
        return cls(
            thread_ts=thread_ts,
            channel_id=channel_id,
            original_channel_id=original_channel_id,
            original_thread_ts=original_thread_ts,
            state=SessionState()
        )

class SlackSessionManager:
    """
    Manager for handling Slack conversation sessions.
    
    This class provides functionality to create, retrieve, update, and delete
    Slack conversation sessions using Redis as a storage backend.
    
    Attributes:
        redis (redis.Redis): Redis client instance
        session_ttl (timedelta): Session time-to-live duration
    """
    
    def __init__(self, redis_url: str, redis_ttl_hours: int) -> None:
        """
        Initialize the session manager.
        
        Args:
            redis_url (str): URL for Redis connection
            redis_ttl_hours (int): Session TTL in hours
        """
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.session_ttl = timedelta(hours=redis_ttl_hours)

    async def get_session(self, channel_id: str, thread_ts: str) -> Optional[SlackSession]:
        """
        Retrieve a session from Redis.
        
        Args:
            channel_id (str): Channel ID
            thread_ts (str): Thread timestamp
            
        Returns:
            Optional[SlackSession]: Session if found, None otherwise
            
        Raises:
            Exception: If there's an error retrieving the session
        """
        try:
            data = await self.redis.get(f"slack:session:{channel_id}:{thread_ts}")
            if data:
                return SlackSession.model_validate_json(data)
            return None
        except Exception as e:
            logger.error(f"Error getting session: {str(e)}")
            return None

    async def create_session(self, channel_id: str, thread_ts: str, original_channel_id: str, original_thread_ts: str) -> SlackSession:
        """
        Create a new session and store it in Redis.
        
        Args:
            channel_id (str): Channel ID
            thread_ts (str): Thread timestamp
            original_channel_id (str): Original channel ID
            original_thread_ts (str): Original thread timestamp
            
        Returns:
            SlackSession: Created session
            
        Raises:
            Exception: If there's an error creating the session
        """
        try:
            session = SlackSession.create(thread_ts, channel_id, original_channel_id, original_thread_ts)
            await self.redis.setex(
                f"slack:session:{channel_id}:{thread_ts}",
                self.session_ttl,
                session.model_dump_json()
            )
            return session
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            raise

    async def update_session(self, session: SlackSession) -> None:
        """
        Update an existing session in Redis.
        
        Args:
            session (SlackSession): Session to update
            
        Raises:
            Exception: If there's an error updating the session
        """
        try:
            await self.redis.setex(
                f"slack:session:{session.channel_id}:{session.thread_ts}",
                self.session_ttl,
                session.model_dump_json()
            )
        except Exception as e:
            logger.error(f"Error updating session: {str(e)}")
            raise

    async def delete_session(self, channel_id: str, thread_ts: str) -> None:
        """
        Delete a session from Redis.
        
        Args:
            channel_id (str): Channel ID
            thread_ts (str): Thread timestamp
            
        Raises:
            Exception: If there's an error deleting the session
        """
        try:
            await self.redis.delete(f"slack:session:{channel_id}:{thread_ts}")
        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}")
            raise

    async def cleanup_expired_sessions(self) -> None:
        """
        Clean up expired sessions.
        
        Note: Redis handles automatic expiration of keys, so this method
        primarily serves as a logging point for cleanup operations.
        
        Raises:
            Exception: If there's an error during cleanup
        """
        try:
            # Redis will automatically handle expiration
            logger.info("Cleanup of expired sessions completed")
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {str(e)}") 
            
    async def close(self) -> None:
        """
        Close the Redis connection.
        
        This method should be called when the session manager is no longer needed
        to properly clean up resources.
        
        Raises:
            Exception: If there's an error closing the connection
        """
        try:
            await self.redis.aclose()
            logger.info("Redis connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {str(e)}")
