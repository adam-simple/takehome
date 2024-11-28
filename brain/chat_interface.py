from models.chat_message import ChatMessage
from models.chat_history import ChatHistory
from models.chat_message_type import ChatMessageType
import dspy
from lms.together import Together
from brain.modules.chatter import Chatter
import logging
from modules.topic_filter import TopicFilterError
from brain.modules.ai_accusation_detector import AIAccusationDetectorError
from datetime import datetime

# import warnings

# warnings.filterwarnings("error")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

lm = Together(
    model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    temperature=0.5,
    max_tokens=1000,
    top_p=0.7,
    top_k=50,
    repetition_penalty=1.2,
    stop=["<|eot_id|>", "<|eom_id|>", "\n\n---\n\n", "\n\n---", "---", "\n---"],
    # stop=["\n", "\n\n"],
)

dspy.settings.configure(lm=lm)
#pyenv
chat_history = ChatHistory()
chatter = Chatter(examples=None)
should_continue = True

while should_continue:
    # Get user input
    user_input = input("You: ")

    # Append user input to chat history
    chat_history.messages.append(
        ChatMessage(
            type=ChatMessageType.FAN,
            datetime=datetime.now(),
            content=user_input,
        ),
    )

    logger.info("--Chat History--")
    logger.info(chat_history)
    logger.info("--End Chat History--")

    # Send request to endpoint
    try:
        response = chatter(chat_history=chat_history).output

        # Append response to chat history
        chat_history.messages.append(
            ChatMessage(
                type=ChatMessageType.CREATOR,
                datetime=datetime.now(),
                content=response,
            ),
        )   

        # Print response
        logger.info(f":::Response {response}")
        print()
        print("Response: ", response)
        print()

    except AIAccusationDetectorError as e:
        logger.error("Fan started to ask probing questions about whether the chatbot is an AI. Exiting.")
        should_continue = False

    except TopicFilterError as e:
        logger.error("Inappropriate topic detected from the chatbot, and no suitable response found. Exiting.")
        should_continue = False

    # uncomment this line to see the 
    lm.inspect_history(n=3)