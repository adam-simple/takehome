import json
import dspy
from dspy.teleprompt import KNNFewShot
from modules.responder import ResponderModule
from models import ChatHistory
from datetime import timedelta


def convert_to_dspy_examples(raw_data):
    examples = []
    for conversation in raw_data:
        chat_history = ChatHistory.model_validate(conversation["chat_history"])
        duration = chat_history.get_conversation_duration()
        if duration is None:
            duration = timedelta(0)
        example = dspy.Example(
            chat_history=chat_history,
            current_time=None,
            conversation_duration=duration,
            output=conversation["output"],
        ).with_inputs("chat_history")
        examples.append(example)
    return examples


def optimize_responder(responder: ResponderModule) -> ResponderModule:
    raw_training_data = json.load(open("training_data/conversations.json"))
    training_examples = convert_to_dspy_examples(raw_training_data)

    optimizer = KNNFewShot(
        k=5, trainset=training_examples, max_bootstrapped_demos=3, max_labeled_demos=3
    )
    compiled_responder = optimizer.compile(responder, trainset=training_examples)

    return compiled_responder
