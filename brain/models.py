from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional


class ChatMessage(BaseModel):
    from_creator: bool
    content: str

    def __str__(self):
        role = "YOU" if self.from_creator else "THE FAN"
        message = role + ": " + self.content
        return message


class ChatHistory(BaseModel):
    messages: List[ChatMessage] = []

    def __str__(self):
        messages = []
        for i, message in enumerate(self.messages):
            message_str = str(message)
            if i == len(self.messages) - 1 and not message.from_creator:
                message_str = "(Most recent message from fan): " + message_str
            messages.append(message_str)
        return "\n".join(messages)

    def model_dump_json(self, **kwargs):
        return str(self)
