# TN Slackbot

A Slack bot that reacts to emoji events using an LLM service.

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your environment variables:
   ```bash
   cp .env.example .env
   ```
5. Update the `.env` file with your Slack and LLM credentials

## Environment Variables

Required environment variables:

- `SLACK_BOT_TOKEN`: Your Slack bot token
- `SLACK_SIGNING_SECRET`: Your Slack app signing secret

Optional environment variables:
- `PORT`: Application port (default: 8000)

## Running the Application

```bash
uvicorn src.main:app --reload
```
