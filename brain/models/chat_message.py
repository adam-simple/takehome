from pydantic import BaseModel
from models.chat_message_type import ChatMessageType
from datetime import datetime

class ChatMessage(BaseModel):
    """
    A single chat message including what type it is and when it was sent.
    """

    type: ChatMessageType
    datetime: datetime
    content: str

    def __str__(self):
        role = self.type.value
        message = role + ": " + self.content
        return message

