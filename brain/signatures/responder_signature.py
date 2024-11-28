import dspy

from brain.models.chat_history import ChatHistory

class ResponderSignature(dspy.Signature):
    """
    Signature for default responder request.
    """

    chat_history: ChatHistory = dspy.InputField(desc="the chat history")

    output: str = dspy.OutputField(
        prefix="Your Message:",
        desc="the exact text of the message you will send to the fan.",
    )