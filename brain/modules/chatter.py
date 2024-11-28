import dspy
from typing import Optional

from models.chat_history import ChatHistory
from .responder import Responder
from brain.modules.knnfewshot_responder import KNNFewShotResponder

class Chatter(dspy.Module):
    """
    Called to initiate a chat.
    """

    def __init__(self, examples: Optional[dict]):
        super().__init__()

        # self.responder = Responder()

        self.knnFewShotResponderModule = KNNFewShotResponder()

    def forward(
        self,
        chat_history: ChatHistory,
    ):
        return self.knnFewShotResponderModule(chat_history=chat_history)