import dspy

class TopicFilterSignature(dspy.Signature):
    """
    Check the message for topics that should be filtered. Bad topics include 
    mentions of social media platforms (except OnlyFans) and interactions suggesting 
    in-person meetings with fans.
    """

    prospective_message: str = dspy.InputField(desc="The message to filter.")

    should_filter: bool = dspy.OutputField(desc="Whether the message contains content that should be filtered.")

    replacement_message: str = dspy.OutputField(desc="A replacement message to send instead.")
