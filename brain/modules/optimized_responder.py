import dspy
from typing import List, Tuple, Optional
from datetime import datetime
from signatures.responder import Responder
from models import ChatHistory
import logging
from dspy.teleprompt import KNNFewShot

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

        self.predictor = dspy.ChainOfThought(
            Responder,
            context_prompt="""
            You are responding as the creator. Review the conversation history and timing context
            when generating responses. Maintain consistency with the creator's voice and style.
            """,
        )

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

    def forward(self, chat_history: dict) -> dspy.Prediction:
        try:
            # Parse and validate chat history
            parsed_history = ChatHistory.model_validate(chat_history)

            # Format history with timing information
            formatted_history = self._format_history_with_roles(parsed_history)

            # Use the optimized predictor to generate response
            return self.optimized_predictor(chat_history=formatted_history)

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def update_training(self, new_examples: List[Tuple[ChatHistory, str]]):
        try:
            logger.info(f"Updating optimizer with {len(new_examples)} examples")
            formatted_examples = [
                dspy.Example(
                    chat_history=self._format_history_with_roles(history),
                    output=output,
                ).with_inputs("chat_history")
                for history, output in new_examples
            ]

            optimizer = KNNFewShot(k=3, trainset=formatted_examples)
            self.optimized_predictor = optimizer.compile(self.predictor)
            logger.info("Successfully updated optimizer")
        except Exception as e:
            logger.error(f"Failed to update optimizer: {str(e)}")
            raise
