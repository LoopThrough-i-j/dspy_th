from datetime import datetime, timezone
import json
import os
from models import ChatMessage, ChatHistory
import dspy
from lms.together import Together

from modules.chatter import ChatterModule

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

chat_history = ChatHistory()

with open(os.path.abspath(os.path.join(os.getcwd(), "training_data/conversations.json")), "r") as file:
    conversation_data = json.load(file)

chatter = ChatterModule(examples=conversation_data)
while True:
    # Get user input
    user_input = input("You: ")

    # Append user input to chat history
    chat_history.messages.append(
        ChatMessage(
            from_creator=False,
            content=user_input,
            message_time=datetime.now(tz=timezone.utc)
        ),
    )

    # Send request to endpoint
    response = chatter(chat_history=chat_history).output

    # Append response to chat history
    chat_history.messages.append(
        ChatMessage(
            from_creator=True,
            content=response,
            message_time=datetime.now(tz=timezone.utc)
        ),
    )
    # Print response
    print()
    print("Response:", response)
    print()
    # uncomment this line to see the 
    # lm.inspect_history(n=1)