from config import MEMORY_PATH

def save_to_memory(role, message):
    with open(MEMORY_PATH, "a", encoding="utf-8") as f:
        f.write(f"{role}: {message}\n")

def load_history():
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""