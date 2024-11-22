import dspy
import logging
from typing import Optional
from models import ChatHistory
from .optimized_responder import OptimizedResponderModule
from data_loader import load_training_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatterModule(dspy.Module):
    """Main module for handling chat interactions."""

    def __init__(self, examples: Optional[dict] = None):
        super().__init__()
        training_examples = load_training_data() if not examples else examples
        self.responder = OptimizedResponderModule(training_examples)

    def forward(self, chat_history: ChatHistory) -> dspy.Prediction:
        """
        Process chat history and generate a safe response.

        Args:
            chat_history: The conversation history

        Returns:
            A prediction containing the filtered response
        """
        try:
            return self.responder(chat_history=chat_history)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Return a safe fallback response
            return dspy.Prediction(
                output="I'm having trouble responding right now. Let's continue our chat on OnlyFans!"
            )
