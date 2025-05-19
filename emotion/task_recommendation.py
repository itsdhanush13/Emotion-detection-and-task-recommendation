def recommend_task(emotion):
    task_mapping = {
        "joy": "Work on a creative brainstorming session.",
        "sadness": "Do light administrative tasks or take a short break.",
        "sad": "Do light administrative tasks or take a short break.",
        "anger": "Engage in a physical activity or mindfulness exercise.",
        "angry": "Engage in a physical activity or mindfulness exercise.",
        "fear": "Review work calmly or discuss concerns with a colleague.",
        "neutral": "Proceed with your regular tasks as planned.",
        "surprise": "Take on a new challenge or learn something different.",
        "disgust": "Take a moment to reset before starting work.",
        "happy": "Try to work on your important & difficult tasks.",
    }

    return task_mapping.get(emotion, "Default work task")