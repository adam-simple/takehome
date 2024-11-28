from datetime import datetime

def current_day_and_time_str() -> str:

    # Get the current datetime
    now = datetime.now()
    
    # Determine the day of the week
    day_of_week = now.strftime("%A")  # Full name of the day (e.g., Monday)
    
    # Determine the rough time of day
    hour = now.hour
    if 5 <= hour < 12:
        time_of_day = "morning"
    elif 12 <= hour < 17:
        time_of_day = "afternoon"
    elif 17 <= hour < 21:
        time_of_day = "evening"
    else:
        time_of_day = "night"
    
    return f"It is currently {time_of_day} on {day_of_week}"