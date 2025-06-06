import os
from dotenv import load_dotenv
load_dotenv()

print("📡 SLACK_BOT_TOKEN:", os.getenv("SLACK_BOT_TOKEN"))
print("📡 SLACK_APP_TOKEN:", os.getenv("SLACK_APP_TOKEN"))

from app.slack_listener import start_slack_listener

if __name__ == "__main__":
    start_slack_listener()
