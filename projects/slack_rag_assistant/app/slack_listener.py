from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os
from app.retriever import handle_question
from app.feedback_handler import build_feedback_blocks

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

app = App(token=SLACK_BOT_TOKEN)


@app.event("app_mention")
def handle_app_mention(event, say, logger):
    logger.info(f" Incoming app_mention: {event}")
    text = event.get("text", "")
    thread_ts = event.get("thread_ts") or event["ts"]

    if not text:
        say(text=" Couldn't extract text.", thread_ts=thread_ts)
        return

    parts = text.split(" ", 1)
    user_input = parts[1] if len(parts) > 1 else ""

    if not user_input:
        say(text=" Please include a question after mentioning me.", thread_ts=thread_ts)
        return

    logger.info(f" Processing input: {user_input}")
    answer = handle_question(user_input)

    say(
        blocks=build_feedback_blocks(answer),
        text=answer,
        thread_ts=thread_ts,
    )


@app.action("feedback_action_positive")
def handle_positive_feedback(ack, body, say, logger):
    ack()
    user = body["user"]["username"]
    logger.info(f" Positive feedback from {user}")
    say(text=f"Thanks for the feedback, {user}!", thread_ts=body["message"]["ts"])


@app.action("feedback_action_negative")
def handle_negative_feedback(ack, body, say, logger):
    ack()
    user = body["user"]["username"]
    logger.info(f" Negative feedback from {user}")
    say(text=f"Sorry to hear that, {user}. We'll try to improve!", thread_ts=body["message"]["ts"])


def start_slack_listener():
    print(" Listening for Slack events...")
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
