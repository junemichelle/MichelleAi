import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    # Tokenize the message
    sentence = tokenize(msg)
    # Convert the tokenized message to bag of words representation
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Pass the bag of words through the model
    output = model(X)
    # Get the predicted tag
    _, predicted = torch.max(output, dim=1)

    # Get the corresponding tag from the list of tags
    tag = tags[predicted.item()]

    # Get the probabilities for each tag
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Check if the predicted tag has a high enough probability
    if prob.item() > 0.75:
        # If yes, find the corresponding intent and return a random response
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == 'funfact':
                    # Special handling for fun facts
                    return random.choice(intent['responses'])
                elif tag == 'languages':
                    # Special handling for languages
                    return random.choice(intent['responses'])
                elif tag == 'talent':
                    # Special handling for talents
                    return random.choice(intent['responses'])
                elif tag == 'name':
                    # Special handling for name
                    return random.choice(intent['responses'])
                elif tag == 'joke':
                    # Special handling for jokes
                    return random.choice(intent['responses'])
                elif tag == 'shy':
                    return random.choice(intent['responses'])
                elif tag == 'mature':
                    return random.choice(intent['responses'])
                elif tag == 'silly':
                    return random.choice(intent['responses'])
                elif tag == 'goodbye':
                    return random.choice(intent['responses'])
                elif tag == 'thanks':
                    return random.choice(intent['responses'])
                elif tag == 'compliment':
                    return random.choice(intent['responses'])
                else:
                    return random.choice(intent['responses'])
    else:
        # If the probability is not high enough, handle general greetings
        for intent in intents['intents']:
            if tag == 'greeting':
                return random.choice(intent['responses'])

    # If no response is found, return a default response
    return "I cannot confirm that, You can however ask her, here is her Instagram account.. trust.me.its.just.me"



if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)