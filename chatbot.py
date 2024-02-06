import random

# Define patterns and responses
patterns = {
    r'hi|hello|hey': ['Hello!', 'Hi there!', 'Hey!'],
    r'how are you': ['I am just a computer program, but I am functioning well. How can I assist you?'],
    r'what is your name': ["I'm a chatbot. You can call me ChatBot."],
    r'bye|goodbye': ['Goodbye!', 'Bye for now.'],
}

# Function to match user input with patterns and generate a response
def respond(user_input):
    for pattern, responses in patterns.items():
        if any(word in user_input.lower() for word in pattern.split('|')):
            return random.choice(responses)
    return "I'm not sure how to respond to that."

print("Hello! I'm your chatbot. Type 'bye' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['bye', 'goodbye']:
        print("ChatBot: Goodbye!")
        break
    response = respond(user_input)
    print("ChatBot:", response)