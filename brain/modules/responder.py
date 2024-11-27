import dspy
from signatures.responder import Responder
from models import ChatHistory
from datetime import datetime, timedelta

class ResponderModule(dspy.Module):
    def __init__(self):
        super().__init__()
        reasoning = dspy.OutputField(
            prefix="Reasoning: Let's think step by step to decide on our message. We"
        )
        self.prog = dspy.TypedChainOfThought(Responder, reasoning=reasoning)

    def forward(self, chat_history: dict, current_time: datetime = None, conversation_duration: timedelta = None):
        chat_history_model = ChatHistory.model_validate(chat_history)
        return self.prog(
            chat_history=chat_history_model,
            current_time=current_time,
            conversation_duration=conversation_duration,
        )