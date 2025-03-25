from slack_bolt import App
import logging

logger = logging.getLogger(__name__)

def handle_david_reaction(slack_app: App, event: dict, say) -> None:
    """
    Handle the David reaction workflow.
    
    Args:
        slack_app: The Slack Bolt app instance
        event: The reaction_added event data
        say: The say function from Slack Bolt
    """
    logger.info("=== Starting handle_david_reaction ===")
    try:
        # Extract event data
        item = event.get("item", {})
        channel = item.get("channel")
        ts = item.get("ts")
        
        if not all([channel, ts]):
            logger.warning("Missing required event data")
            return
        
        # Get all messages in the current conversation
        logger.info("Fetching conversation messages...")
        messages = slack_app.client.conversations_history(
            channel=channel,
            latest=ts,
            limit=100,  # Adjust this number based on your needs
            inclusive=True
        )["messages"]
        
        # Convert messages to string
        messages_str = "\n".join([msg.get("text", "") for msg in messages])
        
        # TODO: Implement similarity search
        # For now, we'll use a placeholder response
        similar_issues_response = "I found some similar issues that might help:\n1. Issue A\n2. Issue B\n3. Issue C"
        
        # Prepare feedback message
        feedback = (
            "Did this help answer your question?\n"
            "React with:\n"
            ":thumbsup: for yes\n" 
            ":thumbsdown: to start the ticket escalation process\n"
        )
        
        # Post response in thread
        response_text = f"{similar_issues_response}\n\n{feedback}"
        say(text=response_text, thread_ts=ts)
        logger.info("Response message sent successfully")
        
    except Exception as e:
        logger.error(f"Error processing David reaction: {str(e)}", exc_info=True)
        say(text="Sorry, I encountered an error while processing your request.", thread_ts=ts)
    finally:
        logger.info("=== Completed handle_david_reaction ===")

def handle_jira_creation_chat(slack_app: App, event: dict, say) -> None:
    """
    Handle the Jira creation chat workflow.
    
    Args:
        slack_app: The Slack Bolt app instance
        event: The reaction_added event data
        say: The say function from Slack Bolt
    """
    logger.info("=== Starting handle_jira_creation_chat ===")
    try:
        # Get the thread timestamp from the original message
        item = event.get("item", {})
        ts = item.get("ts")
        
        if not ts:
            logger.warning("Missing thread timestamp")
            return
            
        say("Complete ticket creation process at http://localhost:8501/", thread_ts=ts)
        logger.info("=== Completed handle_jira_creation_chat ===")  
    except Exception as e:
        logger.error(f"Error in handle_jira_creation_chat: {str(e)}", exc_info=True)

def handle_fraud_noc_ping(slack_app: App, event: dict, say) -> None:
    """
    Handle the trust-and-safety reaction by pinging fraud_noc.
    
    Args:
        slack_app: The Slack Bolt app instance
        event: The reaction_added event data
        say: The say function from Slack Bolt
    """
    logger.info("=== Starting handle_fraud_noc_ping ===")
    try:
        # Get the thread timestamp from the original message
        item = event.get("item", {})
        ts = item.get("ts")
        
        if not ts:
            logger.warning("Missing thread timestamp")
            return
            
        say("@fraud_noc", thread_ts=ts)
        logger.info("=== Completed handle_fraud_noc_ping ===")
    except Exception as e:
        logger.error(f"Error in handle_fraud_noc_ping: {str(e)}", exc_info=True)
