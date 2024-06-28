
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader

# Load intents from JSON file
with open(r"C:\Users\Paru\Downloads\intents.json") as f:
    intents = json.load(f)
import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

sentence = ["hello", "how", "are", "you"]
words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
bog= bag_of_words(sentence,words)
print(bog)
# Extract data from intents
all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = pattern.lower()  # assuming tokenize and preprocessing is done
        all_words.extend(w.split())
        xy.append((w, tag))

ignore_words = ['?', '.', '!']
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(all_words)
print(tags)
# Create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # Bag of words
    bag = []
    pattern_words = pattern_sentence.split()
    for w in all_words:
        bag.append(1) if w in pattern_words else bag.append(0)
    X_train.append(bag)
    # Output label
    y_train.append(tags.index(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)

# Define constants
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
batch_size = 8
learning_rate = 0.001
num_epochs = 1000
print(input_size, output_size)

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_samples = len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples

# Create an instance of ChatDataset
dataset = ChatDataset(X_train, y_train)

# Create DataLoader for batching and shuffling data
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# Instantiate the model
model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
print(f'final loss: {loss.item():.4f}')

print('Training complete!')
# Example of using trained model for prediction
test_sentence = "do you have creatine and how much is the price??"
bag = []
pattern_words = test_sentence.lower().split()
for w in all_words:
    bag.append(1) if w in pattern_words else bag.append(0)
input_data = torch.tensor(np.array(bag), dtype=torch.float32)
input_data = input_data.unsqueeze(0)  # Add batch dimension

# Set model to evaluation mode
model.eval()
with torch.no_grad():
    output = model(input_data)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    print(f"Predicted tag: {tag}")

def get_response(sentence):
    model.eval()
    with torch.no_grad():
        tokenized_sentence = tokenize(sentence)
        X = bag_of_words(tokenized_sentence, all_words)
        X = torch.tensor(X, dtype=torch.float32)
        X = X.unsqueeze(0)
        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]
        for intent in intents['intents']:
            if intent['tag'] == tag:
                response = np.random.choice(intent['responses'])
                return response
            
import tkinter as tk
from tkinter import scrolledtext, Entry, Button

# Function definitions and other imports (chatbot functionality) should remain as per your existing code...

# GUI setup using tkinter
def send_message(event=None):
    user_input = input_box.get()
    if user_input.lower() == "quit":
        window.quit()
    chat_history.insert(tk.END, f"You: {user_input}\n")
    chat_history.see(tk.END)
    bot_response = get_response(user_input)
    chat_history.insert(tk.END, f"Chatbot: {bot_response}\n\n")
    chat_history.see(tk.END)
    input_box.delete(0, tk.END)

window = tk.Tk()
window.title("Gymbot")
window.geometry("800x800")

# Set icon for the window
window.iconphoto(True, tk.PhotoImage(file=r"C:\Users\Paru\OneDrive\Pictures\kitty3.png"))  # Replace with your icon file

# Background Image for the entire window using Canvas
canvas = tk.Canvas(window, width=900, height=600)
canvas.pack()

background_image = tk.PhotoImage(file=r"C:\Users\Paru\Downloads\desktop-wallpaper-gym-cartoon-funny-gym-cartoon-thumbnail.png")  # Replace with your image file
canvas.create_image(0, 0, anchor=tk.NW, image=background_image)

# Fonts
font_style = ("Arial", 12)  # Example font style

# Calculate center position for chatbox
canvas_width = 500
canvas_height = 500
chatbox_width = 100
chatbox_height = 50
x_chatbox = (canvas_width - chatbox_width) / 2
y_chatbox = 50

# Chat History
chat_history = scrolledtext.ScrolledText(canvas, width=40, height=20, bg="light pink", font=font_style)
chat_history.place(x=x_chatbox, y=y_chatbox)

# Display initial message
initial_message = "Hello, I am your personal Gymbot!!"
chat_history.insert(tk.END, initial_message + "\n\n")
chat_history.see(tk.END)

# Input Box
input_box = Entry(canvas, width=30, bg="pink", font=font_style)
input_box.place(x=x_chatbox, y=y_chatbox + chatbox_height + 10)
input_box.bind('<Return>', send_message)

# Send Button
send_button = Button(canvas, text="Send", width=10, command=send_message, bg="light blue", font=font_style)
send_button.place(x=x_chatbox + 250, y=y_chatbox + chatbox_height + 10)

window.mainloop()
