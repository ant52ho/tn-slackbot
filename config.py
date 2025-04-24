import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Log level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Slack Configuration
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_THREAD_FETCH_LIMIT = os.getenv("SLACK_THREAD_FETCH_LIMIT", 100)

# Redis Configuration
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_TTL_HOURS = int(os.getenv("REDIS_TTL_HOURS", 24))
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_URL = os.getenv("REDIS_URL", f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ChromaDB Configuration
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = os.getenv("CHROMA_PORT", 8000)
SIMILAR_ISSUES_COUNT = int(os.getenv("SIMILAR_ISSUES_COUNT", 3))

# Chat Data Path for similarity search
CHAT_DATA_PATH = os.getenv("CHAT_DATA_PATH", "/Users/anthony/Code/tn_slackbot/data/messages2024-04-03-2025-04-10.csv")

# Jira Configuration
JIRA_SERVER = os.getenv("JIRA_SERVER")
JIRA_USERNAME = os.getenv("JIRA_USERNAME")
JIRA_PASSWORD = os.getenv("JIRA_PASSWORD")
JIRA_PROJECT = os.getenv("JIRA_PROJECT")

# app config
SERVER_PORT = int(os.getenv("SERVER_PORT", 8001))

# Validate required environment variables
required_vars = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "OPENAI_API_KEY", "JIRA_SERVER", "JIRA_USERNAME", "JIRA_PASSWORD", "JIRA_PROJECT", "REDIS_PASSWORD"]
missing_vars = [var for var in required_vars if not globals()[var]]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}") 