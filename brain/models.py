from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import List, Optional


class ChatMessage(BaseModel):
    from_creator: bool
    content: str
    timestamp: Optional[datetime] = None

    def __str__(self):
        role = "YOU" if self.from_creator else "THE FAN"
        message = f"{role}: {self.content}"
        return message

class ChatHistory(BaseModel):
    messages: List[ChatMessage] = []

    def __str__(self):
        messages = [str(message) for message in self.messages]
        return "\n".join(messages)

    def get_conversation_duration(self):
        if self.messages:
            if all(message.timestamp is not None for message in self.messages):
                start_time = self.messages[0].timestamp
                end_time = self.messages[-1].timestamp
                return end_time - start_time
        return timedelta(0)
    
    def model_dump_json(self, **kwargs):
        return str(self)