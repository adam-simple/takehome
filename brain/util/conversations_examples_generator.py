import json
from typing import List
from models.chat_history import ChatHistory
from models.chat_message_type import ChatMessageType
from models.chat_message import ChatMessage
from dspy import Example
from datetime import datetime

class ConversationsExamplesGenerator:
    @staticmethod
    def parse_from_file(file_path: str) -> List[Example]:
        with open(file_path, 'r') as file:
            data = json.load(file)

        dspy_examples = []

        for conversation in data:
            chat_messages = []
            for message in conversation.get("chat_history", {}).get("messages", []):
                message_type = ChatMessageType.CREATOR if message.get("from_creator") else ChatMessageType.FAN
                message_datetime = message.get("datetime", datetime.now())
                chat_message = ChatMessage(type=message_type, datetime=message_datetime, content=message.get("content", ""))
                chat_messages.append(chat_message)

            output = conversation.get("output", "")
            chat_history = ChatHistory(messages=chat_messages)

            dspy_examples.append(
                Example(chat_history_example=str(chat_history), output=output).with_inputs("chat_history_example")
            )

        return dspy_examples
