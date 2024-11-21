import dspy
from typing import List, Tuple, Optional
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
        """
        Initialize the optimized responder module.

        Args:
            training_examples: List of (ChatHistory, output) tuples for training
            k: Number of nearest neighbors to use for few-shot examples
        """
        super().__init__()

        # Create predictor with enhanced reasoning
        reasoning = dspy.OutputField(
            prefix="Reasoning: Let's think step by step about how to respond in our client's voice:\n"
            "1. Consider the context and tone of the conversation\n"
            "2. Identify key themes and interests from the chat history\n"
            "3. Formulate a response that matches our client's authentic style\n"
            "We",
        )
        self.predictor = dspy.ChainOfThought(Responder, reasoning=reasoning)

        # Format training examples if provided
        if training_examples:
            try:
                logger.info(
                    f"Initializing optimizer with {len(training_examples)} training examples"
                )
                # Convert training examples to the format KNNFewShot expects
                formatted_examples = [
                    dspy.Example(
                        chat_history=str(history),  # Use ChatHistory's __str__ method
                        output=output,
                    ).with_inputs("chat_history")
                    for history, output in training_examples
                ]

                # Initialize KNNFewShot and compile the predictor
                optimizer = KNNFewShot(k=k, trainset=formatted_examples)
                self.optimized_predictor = optimizer.compile(self.predictor)
                logger.info("Successfully compiled optimizer with training examples")
            except Exception as e:
                logger.error(f"Failed to initialize optimizer: {str(e)}")
                raise
        else:
            # If no training examples, just use the predictor directly
            logger.warning(
                "No training examples provided - will use predictor without optimization"
            )
            self.optimized_predictor = self.predictor

    def forward(self, chat_history: dict) -> dspy.Prediction:
        """
        Generate a response based on the chat history using KNN-retrieved examples.

        Args:
            chat_history: Dictionary containing the conversation history

        Returns:
            dspy.Prediction containing the generated response
        """
        try:
            # Parse and validate chat history
            parsed_history = ChatHistory.parse_obj(chat_history)

            # Use ChatHistory's built-in string representation
            formatted_history = str(parsed_history)

            # Use the optimized predictor to generate response
            return self.optimized_predictor(chat_history=formatted_history)

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def update_training(self, new_examples: List[Tuple[ChatHistory, str]]):
        """
        Update the optimizer with new training examples.

        Args:
            new_examples: List of new (ChatHistory, output) training examples
        """
        try:
            logger.info(f"Updating optimizer with {len(new_examples)} examples")
            # Convert new examples to the format KNNFewShot expects
            formatted_examples = [
                dspy.Example(
                    chat_history=str(history),  # Use ChatHistory's __str__ method
                    output=output,
                ).with_inputs("chat_history")
                for history, output in new_examples
            ]

            # Create new optimizer and compile with new examples
            optimizer = KNNFewShot(k=3, trainset=formatted_examples)
            self.optimized_predictor = optimizer.compile(self.predictor)
            logger.info("Successfully updated optimizer")
        except Exception as e:
            logger.error(f"Failed to update optimizer: {str(e)}")
            raise
