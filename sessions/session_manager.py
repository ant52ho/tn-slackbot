import logging
from datetime import datetime, timedelta
import redis.asyncio as redis
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class TicketData(BaseModel):
    summary: str = ""
    description: str = ""
    priority: str = "P4"
    fraud_noc_pinged: bool = False
    ticket_key: str = ""

class SessionState(BaseModel):
    current_step: str = "initial"
    ticket_data: TicketData = Field(default_factory=TicketData)

class SlackSession(BaseModel):
    thread_ts: str
    channel_id: str
    original_channel_id: str
    original_thread_ts: str
    state: SessionState = Field(default_factory=SessionState)

    @classmethod
    def create(cls, thread_ts: str, channel_id: str, original_channel_id: str, original_thread_ts: str) -> 'SlackSession':
        return cls(
            thread_ts=thread_ts,
            channel_id=channel_id,
            original_channel_id=original_channel_id,
            original_thread_ts=original_thread_ts,
            state=SessionState()
        )

class SlackSessionManager:
    def __init__(self, redis_url: str, redis_ttl_hours: int):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.session_ttl = timedelta(hours=redis_ttl_hours)

    async def get_session(self, channel_id: str, thread_ts: str) -> Optional[SlackSession]:
        """Get session from Redis."""
        try:
            data = await self.redis.get(f"slack:session:{channel_id}:{thread_ts}")
            if data:
                return SlackSession.model_validate_json(data)
            return None
        except Exception as e:
            logger.error(f"Error getting session: {str(e)}")
            return None

    async def create_session(self, channel_id: str, thread_ts: str, original_channel_id: str, original_thread_ts: str) -> SlackSession:
        """Create a new session and store in Redis."""
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
        """Update session in Redis."""
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
        """Delete session from Redis."""
        try:
            await self.redis.delete(f"slack:session:{channel_id}:{thread_ts}")
        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}")
            raise

    async def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        try:
            # Redis will automatically handle expiration
            logger.info("Cleanup of expired sessions completed")
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {str(e)}") 
            
    async def close(self) -> None:
        """Close the Redis connection."""
        try:
            await self.redis.aclose()
            logger.info("Redis connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {str(e)}")
