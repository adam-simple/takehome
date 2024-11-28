from typing import List
import dspy
import logging

from util.current_day_and_time import current_day_and_time_str
from signatures.knnfewshot_responder_signature import KNNFewShotResponderSignature
from models.chat_history import ChatHistory
from dspy.teleprompt import KNNFewShot
from brain.util.conversations_examples_generator import ConversationsExamplesGenerator
from modules.topic_filter import TopicFilter, TopicFilterError
from brain.modules.ai_accusation_detector import AIAccusationDetector, AIAccusationDetectorError

class KNNFewShotResponder(dspy.Module):
    """
    Slightly more elaborate responder class that uses K-Nearest Neighbour Fewshot functionality
    to better simulate a creator's voice.
    """

    def __init__(self):
        super().__init__()

        reasoning = dspy.OutputField(
            prefix="Reasoning: Let's think step by step to decide on our message. We",
        )

        self.topic_filter = TopicFilter()
        self.ai_accusation_detector = AIAccusationDetector()

        dspy_examples = ConversationsExamplesGenerator.parse_from_file('training_data/conversations.json')
        self.knnfewshot_teleprompter = KNNFewShot(k=1, trainset=dspy_examples)
    
    def forward(
        self,
        chat_history: ChatHistory,
    ):
        
        # First we detect any AI accusations from the user.
        logging.info(":::Detecting AI accusations.")
        ai_accusation_detected = self.ai_accusation_detector(chat_history=chat_history)
        if ai_accusation_detected:
            logging.info(":::Raising exception.")
            raise AIAccusationDetectorError("This fan is starting to get suspicious. Notify creator.")

        # Then generate the response using the KNNFewShot functionality.
        logging.info(":::Generating response.")
        predictor = dspy.ChainOfThought(KNNFewShotResponderSignature)
        self.compiled_knnfewshot_teleprompter = self.knnfewshot_teleprompter.compile(predictor)

        current_day_and_time = current_day_and_time_str()
        chat_description = chat_history.description()

        response = self.compiled_knnfewshot_teleprompter(
            chat_history=str(chat_history),
            current_day_and_time=current_day_and_time,
            chat_description=chat_description
        )
        
        logging.info(f":::Generated response: {response.output}")

        # Now we check for topics that should be filtered.
        logging.info(":::Topic filtering.")
        should_filter, suggested_replacement = self.topic_filter(response.output)

        # If we have any bad topics, we either use the replacement or raise an exception.
        if should_filter:
            if suggested_replacement:
                logging.info(":::Replacing with filter suggestion..")
                response.output = suggested_replacement
            else:
                logging.info(":::Raising exception.")
                raise TopicFilterError("No suitable suggested replacement found. Notify creator.")
            
        # At this point everything is ok, so return the response.
        logging.info(":::Successully generated response.")
        return response
