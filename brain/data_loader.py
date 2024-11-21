import json
from pathlib import Path
from typing import List, Tuple
from models import ChatHistory, ChatMessage


def load_training_data(
    file_path: str = "training_data/conversations.json",
) -> List[Tuple[ChatHistory, str]]:
    """
    Load and parse training data from a JSON file into a list of (ChatHistory, output) tuples.

    Args:
        file_path (str): Path to the JSON file containing training conversations

    Returns:
        List[Tuple[ChatHistory, str]]: List of (chat_history, output) pairs

    Raises:
        FileNotFoundError: If the training data file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        ValueError: If the data format is unexpected
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Training data file not found: {file_path}")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Training data must be a list of examples")

        # Convert the raw data into proper ChatHistory objects
        training_examples = []
        for example in data:
            if (
                not isinstance(example, dict)
                or "chat_history" not in example
                or "output" not in example
            ):
                raise ValueError(
                    "Each training example must have 'chat_history' and 'output' fields"
                )

            chat_history = ChatHistory.parse_obj(example["chat_history"])
            output = example["output"]
            training_examples.append((chat_history, output))

        return training_examples

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in training data file: {e}", e.doc, e.pos
        )
