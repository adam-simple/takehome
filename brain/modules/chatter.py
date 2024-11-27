import dspy
from typing import Optional
from datetime import datetime

from models import ChatHistory
from modules.responder import ResponderModule
from modules.filter import FilterModule
from optimize import optimize_responder


class ChatterModule(dspy.Module):
    def __init__(self, examples: Optional[dict]):
        super().__init__()
        self.responder = optimize_responder(ResponderModule())
        self.filter = FilterModule()

    def forward(
        self,
        chat_history: ChatHistory,
    ):
        # First check if the last message is appropriate
        if chat_history.messages:
            filter_result = self.filter(chat_history=chat_history)
            if not filter_result.is_appropriate:
                return dspy.Prediction(output=filter_result.suggested_response)

        # If message is appropriate, proceed with normal response
        current_time = datetime.now()
        conversation_duration = chat_history.get_conversation_duration()
        return self.responder(
            chat_history=chat_history,
            current_time=current_time,
            conversation_duration=conversation_duration,
        )
