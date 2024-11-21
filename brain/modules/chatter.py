import dspy
from typing import Optional
from models import ChatHistory
from .optimized_responder import OptimizedResponderModule
from data_loader import load_training_data


class ChatterModule(dspy.Module):
    """
    A module that manages chat interactions using an optimized responder
    trained on example conversations.
    """

    def __init__(self, examples: Optional[dict] = None):
        super().__init__()
        # Load training examples from file if not provided directly
        training_examples = examples if examples is not None else load_training_data()
        self.responder = OptimizedResponderModule(training_examples=training_examples)

    def forward(
        self,
        chat_history: ChatHistory,
    ):
        """
        Generate a response based on the chat history using the optimized responder.

        Args:
            chat_history: The conversation history to respond to

        Returns:
            The generated response from the optimized responder
        """
        return self.responder(chat_history=chat_history)
