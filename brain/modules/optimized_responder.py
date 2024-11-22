import dspy
from typing import List, Tuple, Optional
from datetime import datetime
from signatures.responder import Responder
from models import ChatHistory
from .topic_filter import TopicFilter
import logging
from dspy.teleprompt import KNNFewShot
from signatures.emotion import EmotionSignature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedResponderModule(dspy.Module):
    """
    A DSPy module that uses KNNFewShot optimization to generate responses that match
    the client's voice and style based on training examples.
    """

    def __init__(
        self,
        training_examples: Optional[List[Tuple[ChatHistory, str]]] = None,
        k: int = 3,
    ):
        super().__init__()

        # Initialize topic filter
        self.topic_filter = TopicFilter()

        self.predictor = dspy.ChainOfThought(
            Responder,
            context_prompt="""
            You are responding as the creator. Review the conversation history and timing context
            when generating responses. Maintain consistency with the creator's voice and style.
            """,
        )

        # Initialize emotion detector
        self.emotion_detector = dspy.ChainOfThought(EmotionSignature)

        # Format training examples if provided
        if training_examples:
            try:
                logger.info(
                    f"Initializing optimizer with {len(training_examples)} training examples"
                )
                formatted_examples = [
                    dspy.Example(
                        chat_history=self._format_history_with_roles(history),
                        output=output,
                    ).with_inputs("chat_history")
                    for history, output in training_examples
                ]

                # Initialize KNNFewShot with trainset and compile the predictor
                optimizer = KNNFewShot(k=k, trainset=formatted_examples)
                self.optimized_predictor = optimizer.compile(self.predictor)
                logger.info("Successfully compiled optimizer with training examples")
            except Exception as e:
                logger.error(f"Failed to initialize optimizer: {str(e)}")
                raise
        else:
            logger.warning(
                "No training examples provided - will use predictor without optimization"
            )
            self.optimized_predictor = self.predictor

    def _format_history_with_roles(self, chat_history: ChatHistory) -> str:
        """Format chat history with explicit role labels and timing information."""
        formatted_messages = []
        for msg in chat_history.messages:
            role = "Creator" if msg.from_creator else "Fan"
            time_str = msg.timestamp.strftime("%I:%M %p")
            formatted_messages.append(f"[{time_str}] {role}: {msg.content}")
        return "\n".join(formatted_messages)

    def _get_recent_responses(self, chat_history: ChatHistory) -> List[str]:
        """Get recent responses from the creator to avoid repetition."""
        recent_responses = []
        for msg in reversed(chat_history.messages):
            if msg.from_creator:
                recent_responses.append(msg.content)
            if len(recent_responses) >= 3:  # Get last 3 creator responses
                break
        return recent_responses

    def _analyze_emotion(self, message: str) -> dict:
        """Analyze emotional content of a message."""
        try:
            result = self.emotion_detector(message=message)
            return {
                "emotion": result.detected_emotion,
                "intensity": result.intensity,
                "recommended_tone": result.recommended_tone,
            }
        except Exception as e:
            logger.error(f"Error analyzing emotion: {str(e)}")
            return {
                "emotion": "neutral",
                "intensity": 3,
                "recommended_tone": "balanced",
            }

    def forward(self, chat_history: dict) -> dspy.Prediction:
        try:
            # Parse and validate chat history
            parsed_history = ChatHistory.model_validate(chat_history)

            # Get the fan's last message
            last_fan_message = next(
                (
                    msg
                    for msg in reversed(parsed_history.messages)
                    if not msg.from_creator
                ),
                None,
            )

            # Analyze emotion if there's a fan message
            emotion_context = ""
            if last_fan_message:
                emotion_info = self._analyze_emotion(last_fan_message.content)
                emotion_context = (
                    f"\nThe fan's message shows {emotion_info['emotion']} "
                    f"emotion with intensity {emotion_info['intensity']}. "
                    f"Respond with a {emotion_info['recommended_tone']} tone."
                )

            # Format history with timing information
            formatted_history = self._format_history_with_roles(parsed_history)

            # Get recent responses to avoid repetition
            recent_responses = self._get_recent_responses(parsed_history)

            # Add recent responses and emotion context
            context_with_history = (
                f"{formatted_history}\n\n"
                f"Your recent responses were:\n"
                f"{chr(10).join(recent_responses)}\n\n"
                f"Please provide a response that maintains your voice while avoiding "
                f"repeating similar phrases or ideas from your recent messages."
                f"{emotion_context}"
            )

            # Generate response with enhanced context
            response = self.optimized_predictor(chat_history=context_with_history)

            # Check if response is safe using topic filter
            is_safe, reasoning, suggested_fix = self.topic_filter(response.output)

            if not is_safe:
                logger.info(f"Response filtered. Reason: {reasoning}")
                if suggested_fix:
                    response.output = suggested_fix
                else:
                    response.output = "I'd love to continue our chat here on OnlyFans! What would you like to talk about?"

            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
