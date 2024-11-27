import dspy
from signatures.filter import TopicFilter
from models import ChatHistory


class FilterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(TopicFilter)

    def forward(self, chat_history: dict):
        return self.prog(chat_history=ChatHistory.model_validate(chat_history))
