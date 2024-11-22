from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional


class ChatMessage(BaseModel):
    from_creator: bool
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

    def __str__(self):
        role = "YOU" if self.from_creator else "THE FAN"
        time_str = self.timestamp.strftime("%I:%M %p")
        return f"[{time_str}] {role}: {self.content}"


class ChatHistory(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.now)

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

    def get_duration(self) -> float:
        """Returns conversation duration in minutes"""
        if not self.messages:
            return 0.0
        latest = self.messages[-1].timestamp
        return (latest - self.start_time).total_seconds() / 60
