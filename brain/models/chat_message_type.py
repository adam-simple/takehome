from enum import Enum

class ChatMessageType(Enum):
    """
    Type of chat message, currently only creator/fan.
    """
    
    CREATOR = "CREATOR"
    FAN = "FAN"

    def __str__(self):
        return "YOU" if self == ChatMessageType.CREATOR else "FAN"