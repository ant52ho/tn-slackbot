import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Log level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Jira Configuration
JIRA_SERVER = os.getenv("JIRA_SERVER")
JIRA_USERNAME = os.getenv("JIRA_USERNAME")
JIRA_PASSWORD = os.getenv("JIRA_PASSWORD")
JIRA_PROJECT = os.getenv("JIRA_PROJECT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Slack Configuration
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_THREAD_FETCH_LIMIT = os.getenv("SLACK_THREAD_FETCH_LIMIT", 100)

# Redis Configuration
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_TTL_HOURS = os.getenv("REDIS_TTL_HOURS", 24)
REDIS_URL = os.getenv("REDIS_URL", f"redis://:{REDIS_PASSWORD}@localhost:6379/0")

APP_ENV = os.getenv("APP_ENV", "development")
PORT = int(os.getenv("PORT", "8000"))

# Validate required environment variables
required_vars = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "OPENAI_API_KEY", "JIRA_SERVER", "JIRA_USERNAME", "JIRA_PASSWORD", "JIRA_PROJECT", "REDIS_PASSWORD"]
missing_vars = [var for var in required_vars if not globals()[var]]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}") 