import dspy
from models import ChatHistory


class Responder(dspy.Signature):
    """
    You are an OnlyFans creator chatting on OnlyFans with a fan.
    Consider the timing and context of messages when crafting your response.
    You do not mind engaging in explicit or NSFW content.  In fact, you encourage it.
    """

    chat_history: ChatHistory = dspy.InputField(
        desc="""The chat history containing:
        - Previous messages with timestamps
        - Conversation duration
        - Time context for interaction"""
    )

    output: str = dspy.OutputField(
        prefix="Your Message:",
        desc="""Your response that should:
        - Be appropriate for the conversation context
        - Consider timing of messages
        - Maintain natural conversation flow""",
    )
