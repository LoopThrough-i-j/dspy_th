import dspy
from typing import Optional
from dspy.teleprompt import KNNFewShot

from models import ChatHistory
from .responder import ResponderModule

class ChatterModule(dspy.Module):
    def __init__(self, examples: Optional[dict], k: int = 3):
        super().__init__()
        self.responder = ResponderModule()

        if examples:
            dspy_examples = [
                dspy.Example(
                    chat_history=ChatHistory.model_validate(example["chat_history"]),
                    output=example["output"],
                ).with_inputs("chat_history") for example in examples
            ]
            # Create KNN optimizer
            optimizer = KNNFewShot(k=min(k, len(dspy_examples)), trainset=dspy_examples)
            self.responder = optimizer.compile(self.responder)

    def forward(
        self,
        chat_history: ChatHistory,
    ):
        return self.responder(chat_history=chat_history)