"""Role extraction helpers for Bespoke-Stratos-17k style records."""


def extract_roles(obj):
    """Extract system, first user, and first assistant messages."""
    system_prompt = obj.get("system", "") or ""
    question = None
    assistant = None

    for message in obj.get("conversations", []):
        role = message.get("from", "")
        value = message.get("value", "") or ""
        if role == "user" and question is None:
            question = value
        elif role == "assistant" and assistant is None:
            assistant = value

    return system_prompt, question or "", assistant or ""
