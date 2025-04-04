import asyncio
import os
import sys
from config import REDIS_URL
import redis.asyncio as redis
import json
from datetime import datetime

async def test_redis_connection():
    """Test basic Redis connectivity and operations"""
    print(f"Testing Redis connection with URL: {REDIS_URL}")
    
    try:
        # Connect to Redis
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        
        # Test basic operations
        print("Testing basic Redis operations...")
        
        # Set a test key
        test_key = "test:connection"
        test_value = {"timestamp": datetime.now().isoformat(), "status": "connected"}
        await redis_client.set(test_key, json.dumps(test_value))
        print(f"✅ Set key '{test_key}' successfully")
        
        # Get the test key
        retrieved_value = await redis_client.get(test_key)
        print(f"✅ Retrieved key '{test_key}' successfully")
        print(f"Retrieved value: {retrieved_value}")
        
        # Delete the test key
        await redis_client.delete(test_key)
        print(f"✅ Deleted key '{test_key}' successfully")
        
        # Test session manager operations
        print("\nTesting session manager operations...")
        from sessions.session_manager import SlackSessionManager, SlackSession
        
        session_manager = SlackSessionManager(REDIS_URL, 1)
        
        # Create a test session
        test_thread_ts = "test_thread_123"
        test_channel_id = "test_channel_123"

        session = await session_manager.create_session(test_channel_id, test_thread_ts)
        print(f"✅ Created session for thread {test_thread_ts}")
        print(f"Session state: {session}")
        
        # Retrieve the session
        retrieved_session = await session_manager.get_session(test_channel_id, test_thread_ts)
        print(f"✅ Retrieved session for thread {test_thread_ts}")
        print(f"Session state: {retrieved_session}")
        
        # Update the session
        retrieved_session.state.current_step = "updated"
        await session_manager.update_session(retrieved_session)
        print(f"✅ Updated session for thread {test_thread_ts}")
        print(f"Session state: {retrieved_session}")
        
        # Delete the session
        await session_manager.delete_session(test_channel_id, test_thread_ts)
        print(f"✅ Deleted session for thread {test_thread_ts}")
        
        # Verify deletion
        deleted_session = await session_manager.get_session(test_channel_id, test_thread_ts)
        if deleted_session is None:
            print(f"✅ Verified session was deleted")
        else:
            print(f"❌ Session was not deleted properly")
        
        print("\n✅ All Redis tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Redis test failed: {str(e)}")
        return False
    finally:
        # Close the Redis connection
        await redis_client.aclose()  # Using aclose() instead of close() as per deprecation warning

if __name__ == "__main__":
    asyncio.run(test_redis_connection()) 