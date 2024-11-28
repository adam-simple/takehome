from typing import Tuple
import dspy

from signatures.topic_filter_signature import TopicFilterSignature

class TopicFilterError(Exception):
    """Error indicating a problem with the topic filter."""
    def __init__(self, message):
        super().__init__(message)

class TopicFilter(dspy.Module):
    """
    Filters messages for topics we don't want to see.
    """

    def __init__(self):
        super().__init__()
        self.filter = dspy.ChainOfThought(TopicFilterSignature)

    def forward(self, prospective_message: str) -> Tuple[bool, str]:

        result = self.filter(prospective_message=prospective_message)

        should_filter_bool = result.should_filter.strip().lower() in {"true", "yes"}

        return should_filter_bool, result.replacement_message
    