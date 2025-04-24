# TN Slackbot

A Slack bot that reacts to emoji events using an LLM service.

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python3 -m venv .slackbot-env
   source .slackbot-env/bin/activate  # On Windows: slackbot-env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your environment variables:
   ```bash
   cp .env.example .env
   ```
5. Update the `.env` file with your Slack and LLM credentials

## Running the Application

```bash
docker compose up -d
```

# First time setup:

## Loading the data (should take ~10 mins)

```bash
python3 utils/test_chat_dataloader.py
```

If this doesn't work, inspect the program to ensure you are loading to the correct location.

## Run the main program

```bash
uvicorn main:fastapi_app --host 0.0.0.0 --port 8001 --reload
```
