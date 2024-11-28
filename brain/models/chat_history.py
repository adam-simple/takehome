from pydantic import BaseModel
from typing import List
from .chat_message import ChatMessage
from datetime import timedelta

class ChatHistory(BaseModel):
    """
    A collection of chat messages.
    """
    
    messages: List[ChatMessage] = []

    def __str__(self):
        return "\n".join(str(message) for message in self.messages)

    def conversation_duration(self) -> timedelta:
        if not self.messages:
            raise ValueError("ChatHistory has no messages to calculate a time delta.")
        
        # Extract timestamps from messages
        timestamps = [message.timestamp for message in self.messages]
        
        # Calculate the delta
        oldest = min(timestamps)
        newest = max(timestamps)
        return newest - oldest

    def conversation_duration_human_readable(self) -> str:
        conversation_duration = self.conversation_duration()

        days = conversation_duration.days
        seconds = conversation_duration.seconds
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days} day{'s' if days > 1 else ''}")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
        if seconds > 0:
            parts.append(f"{seconds} second{'s' if seconds > 1 else ''}")
        
        return ", ".join(parts) if parts else "less than a second"
    
    def description(self) -> str:
        return f"This conversation has been going on for {self.conversation_duration_human_readable} and contains {self.messages.count} messages."

    def model_dump_json(self, **kwargs):
        return str(self)
