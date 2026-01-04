def build_feedback_blocks(answer: str):
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{answer}"
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Helpful"
                    },
                    "value": "positive_feedback",
                    "action_id": "feedback_action_positive"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Not Helpful"
                    },
                    "value": "negative_feedback",
                    "action_id": "feedback_action_negative"
                }
            ]
        }
    ]
