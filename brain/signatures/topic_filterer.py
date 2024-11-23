import dspy

from models import ChatHistory

class TopicFilterer(dspy.Signature):
    """
    Detect and gracefully divert the conversation if users chat is about 
    social profiles other than only fans, in-person meetups or personal information.
    """

    chat_history: ChatHistory = dspy.InputField(desc="the chat history")

    is_safe: bool = dspy.OutputField(
        desc="Respond with true or false",
    )
    output: str = dspy.OutputField(
        prefix="Your Message:",
        desc="The diverted message sent to the Fan",
    )