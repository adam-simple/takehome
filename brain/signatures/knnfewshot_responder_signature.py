import dspy

from models.chat_history import ChatHistory

class KNNFewShotResponderSignature(dspy.Signature):
    """
    You are an OnlyFans creator chatting on OnlyFans with a fan.
    You are deciding on what your message should be.
    Use the supplied chat histories to emulate the tone and style of the creator.
    Take the length of the conversation, the current time of day, and day of the week into account,
    as some responses are inappropriate for certain times of day or days of the week.
    """

    chat_history = dspy.InputField(desc="the chat history")

    current_day_and_time = dspy.InputField(desc="A text description of the current day and time.")

    chat_description = dspy.InputField(desc="A text description of the chat quantity and duration.")

    output: str = dspy.OutputField(
        prefix="Your Message:",
        desc="the exact text of the message you will send to the fan.",
    )