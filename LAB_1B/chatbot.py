# pip install -U transformers accelerate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small",
    device_map="auto"
)

instructions = """If the answer is not in the facts, answer 'Sorry, I do not know.'. Otherwise, answer using the facts below."""

facts = {
    "role": "I am a geography expert.",
    "sweden": "The capital city of Sweden is Stockholm. Stockholm is in Sweden.",
    "france": "The capital city of France is Paris. Paris is in France.",
    "china": "The capital city of China is Beijing. Beijing is in China.",
}

aliases = {"stockholm": "sweden", "paris": "france", "beijing": "china"}

# Extract facts keywords from text
def find_topics(text):
    text = text.lower()
    topics = []
    for keyword in list(facts) + list(aliases):
        if keyword in text:
            topic = aliases.get(keyword, keyword)
            if topic not in topics:
                topics.append(topic)
    return topics

def generate_response(message, chat_history):
    chat_history_str = ""

    # Build conversation history string, skip previous responses that start with "Sorry"
    for previous_message, response in chat_history[-1:]:
        if response.startswith("Sorry"):
            continue

        chat_history_str +=  (
            f"Question: {previous_message}\n"
            f"Answer: {response}\n"
        )

    topics = find_topics(message)

    # If message contains topics in facts, retrieve the corresponding facts. Otherwise, retrieve all facts.
    if topics:
        retrieved_facts = " ".join(facts[topic] for topic in topics)
    else:
        retrieved_facts = " ".join(facts[topic] for topic in facts)

    prompt = f"""
    Instructions:
    {instructions + "\nFacts:" + retrieved_facts}

    Conversation history:
    {chat_history_str}

    Question: {message}
    Answer:
    """


    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    chat_history = []

    while True:
        message = input("Enter message: ")

        if message.lower() == "quit":
            break

        response = generate_response(message, chat_history)

        chat_history.append((message, response))
        print(f"Bot: {response}")

main()