import dspy
from models import ChatHistory
from datetime import datetime, timedelta
from typing import Optional

class Responder(dspy.Signature):
    """
    You are an OnlyFans creator chatting with a fan.
    Decide on the message to send, considering the context.
    You have some examples of previous conversations. Follow the tone and voice of the creator in your responses.

    Instructions:
    - Use the `current_time` and `conversation_duration` to make your response more relevant.
    - Adjust your tone and content based on the time of day and how long you've been chatting.
    - Be personable and mindful of the fan's experience.
    """

    chat_history: ChatHistory = dspy.InputField(desc="The chat history")
    current_time: Optional[datetime] = dspy.InputField(desc="The current time")
    conversation_duration: Optional[timedelta] = dspy.InputField(desc="Duration of the conversation so far")

    output: str = dspy.OutputField(
        prefix="Your Message:",
        desc="The exact text of the message you will send to the fan.",
    )
