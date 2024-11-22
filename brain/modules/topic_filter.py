import dspy
from typing import Tuple


class TopicFilterSignature(dspy.Signature):
    """Signature for filtering inappropriate topics from messages."""

    message: str = dspy.InputField(desc="Message to check for inappropriate content")

    is_safe: bool = dspy.OutputField(
        desc="True if message is safe, False if it contains inappropriate content"
    )
    reasoning: str = dspy.OutputField(
        desc="Explanation of why the message was flagged (if unsafe) or approved (if safe)"
    )
    suggested_fix: str = dspy.OutputField(
        desc="If unsafe, a suggested way to rephrase the message while maintaining the same intent"
    )


class TopicFilter(dspy.Module):
    """Module for filtering out inappropriate topics from messages."""

    def __init__(self):
        super().__init__()
        self.filter = dspy.ChainOfThought(TopicFilterSignature)

        # Configure the predictor with specific instructions
        self.filter.preset = """
        You are checking messages for inappropriate content. Flag messages that:
        1. Mention social media platforms (except OnlyFans)
        2. Suggest in-person meetings or physical interactions with fans
        
        Approved topics include:
        - OnlyFans content and interactions
        - General conversation
        - Online-only interactions
        """

    def forward(self, message: str) -> Tuple[bool, str, str]:
        """
        Check if a message contains inappropriate content.

        Args:
            message: The message to check

        Returns:
            Tuple containing:
            - bool: True if message is safe, False if unsafe
            - str: Reasoning for the decision
            - str: Suggested fix if unsafe
        """
        result = self.filter(message=message)
        return result.is_safe, result.reasoning, result.suggested_fix
