import dspy

class AIAccusationDetectorSignature(dspy.Signature):
    """
    Check the message for whether it contains accusations, questions, or indications from the fan that they believe the creator is actually an AI.
    """

    chat_history: str = dspy.InputField(desc="The chat history to inspect.")

    accusation_present: bool = dspy.OutputField(desc="Whether the chat history contains accusations.")
