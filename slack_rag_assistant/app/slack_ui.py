from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

class SlackUI:
    def __init__(self, token):
        self.client = WebClient(token=token)

    def create_message_block(self, text, button_text, button_action_id):
        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": text
            },
            "accessory": {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": button_text
                },
                "action_id": button_action_id
            }
        }

    def send_message(self, channel, blocks):
        try:
            response = self.client.chat_postMessage(
                channel=channel,
                blocks=blocks
            )
            return response
        except SlackApiError as e:
            print(f"Error sending message: {e.response['error']}")

    def format_response(self, response_text):
        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": response_text
            }
        }