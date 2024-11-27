import dspy
from models import ChatHistory


class TopicFilter(dspy.Signature):
    """
    Determine if a message is appropriate and provide smart handling of filtered content.
    """

    chat_history: ChatHistory = dspy.InputField(
        desc="The chat history to analyze",
        format=lambda x: str(x),
    )

    is_appropriate: bool = dspy.OutputField(desc="Whether the message is appropriate")
    reasoning: str = dspy.OutputField(desc="Explanation for the filtering decision")
    suggested_response: str = dspy.OutputField(
        desc="A suggested alternative response that maintains creator voice while redirecting the conversation appropriately"
    )
