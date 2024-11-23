import dspy

from brain.signatures.topic_filterer import TopicFilterer
from signatures.responder import Responder
from models import ChatHistory

class ResponderModule(dspy.Module):
    def __init__(self):
        super().__init__()
        reasoning = dspy.OutputField(
            prefix="Reasoning: Let's think step by step to decide on our message. We",
        )
        self.chat_filter = dspy.TypedPredictor(TopicFilterer)
        self.prog = dspy.TypedChainOfThought(Responder, reasoning=reasoning)
    
    def forward(
        self,
        chat_history: ChatHistory,
    ):
        filtered_data = self.chat_filter(chat_history=chat_history)
        return self.prog(
            chat_history=chat_history,
        ) if filtered_data.is_safe else filtered_data
