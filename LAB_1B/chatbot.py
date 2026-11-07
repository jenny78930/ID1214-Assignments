# pip install -U transformers accelerate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small",
    device_map="auto"
)

instructions = """You role is a geographic expert.
            Answer questions about capital cities accurately.
            Use the conversation history when answering follow-up questions.
            Keep your answers short and clear.
            The capital city of Sweden is Stockholm. Stockholm is in Sweden.
            The capital city of France is Paris. Paris is in France."""

def generate_response(message, chat_history):
    chat_history_str = ""

    for previous_message, response in chat_history[-2:]:
        chat_history_str +=  (
            f"User: {previous_message}\n"
            f"Assistant: {response}\n"
        )

    prompt = f"""
    Instructions:
    {instructions}

    Conversation history:
    {chat_history_str}

    User: {message}
    Assistant:
    """


    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    chat_history = []

    while True:
        message = input("Enter message: ")

        print(chat_history)

        if message.lower() == "quit":
            break

        response = generate_response(message, chat_history)

        chat_history.append((message, response))

        print(f"Bot: {response}")

main()