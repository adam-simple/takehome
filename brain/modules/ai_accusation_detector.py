from typing import Tuple
import dspy
from models.chat_history import ChatHistory

from signatures.ai_accusation_detector_signature import AIAccusationDetectorSignature

class AIAccusationDetectorError(Exception):
    """Error indicating the fan is starting to ask too many questions."""
    def __init__(self, message):
        super().__init__(message)

class AIAccusationDetector(dspy.Module):
    """
    Fans are suspicious creatures and sometimes their shameful lack of faith might mean they start
    digging a little too deep into the truth. Until goons are in the budget, this will have to do
    as a stopgap measure.
    """

    def __init__(self):
        super().__init__()
        self.filter = dspy.ChainOfThought(AIAccusationDetectorSignature)

        # self.filter.preset = """
        #     Check the message for accusations or questions from the fan about whether 
        #     the creator is actually an AI.
        #     """

    def forward(self, chat_history: ChatHistory) -> Tuple[bool, str]:

        result = self.filter(chat_history=str(chat_history))

        # Seems to produce strings even though the type is Bool. Weird. Probably an 
        # amazingly good reason for it.
        return result.accusation_present.strip().lower() in {"true", "yes"}
