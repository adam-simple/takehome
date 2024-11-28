import dspy

from brain.signatures.responder_signature import ResponderSignature
from brain.models.chat_history import ChatHistory

class Responder(dspy.Module):
    """
    Default unoptimised responder module.
    """

    def __init__(self):
        super().__init__()
        reasoning = dspy.OutputField(
            prefix="Reasoning: Let's think step by step to decide on our message. We",
        )
        self.prog = dspy.ChainOfThought(ResponderSignature, reasoning=reasoning)
    
    def forward(
        self,
        chat_history: dict,
    ):
        return self.prog(
            chat_history=ChatHistory.parse_obj(chat_history),
        )