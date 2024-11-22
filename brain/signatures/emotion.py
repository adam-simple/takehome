import dspy


class EmotionSignature(dspy.Signature):
    """Signature for detecting emotion in messages."""

    message: str = dspy.InputField(desc="Message to analyze for emotional content")

    detected_emotion: str = dspy.OutputField(
        desc="Primary emotion detected (e.g., happy, sad, excited, anxious, neutral)"
    )

    intensity: int = dspy.OutputField(
        desc="Emotion intensity on scale 1-5, where 1 is subtle and 5 is intense"
    )

    recommended_tone: str = dspy.OutputField(
        desc="Recommended tone for response (e.g., supportive, enthusiastic, calming)"
    )
