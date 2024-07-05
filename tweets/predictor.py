import os
import torch
import torch.nn as nn
import re
import json

class SimpleCNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, kernel_sizes, num_filters):
        super(SimpleCNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.conv = nn.Conv2d(1, num_filters, (3, embed_size))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = torch.relu(self.conv(x)).squeeze(3)
        x = torch.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

vocab_path = os.path.join(BASE_DIR, 'model', 'vocab.json')
with open(vocab_path, 'r') as f:
    vocab = json.load(f)

vocab_size = len(vocab) + 1  

model_path = os.path.join(BASE_DIR, 'model', 'cnn_text_classifier.pth')
model = SimpleCNNTextClassifier(vocab_size=vocab_size, embed_size=128, num_classes=1, kernel_sizes=[3], num_filters=100)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Added map_location for CPU loading
model.eval()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'@\\w+', '', text)
    text = re.sub(r'\\d+', '', text)
    text = re.sub(r'\\s+', ' ', text)
    text = re.sub(r'[^\\w\\s]', '', text)
    return text.strip()

def word_tokenize(text):
    return re.findall(r'\\b\\w+\\b', text)

def text_to_sequence(text, vocab, max_seq_length=100):
    tokens = word_tokenize(preprocess_text(text))
    sequence = [vocab.get(word, 0) for word in tokens]
    if len(sequence) < max_seq_length:
        sequence += [0] * (max_seq_length - len(sequence))
    return sequence[:max_seq_length]

def is_abusive(content, model, vocab, max_text_length=1000):
    if len(content) > max_text_length:
        raise ValueError(f"Input text length exceeds maximum allowed length of {max_text_length}")
    
    sequence = text_to_sequence(content, vocab)
    with torch.no_grad():
        output = model(torch.tensor([sequence], dtype=torch.long))
        prediction = torch.sigmoid(output).item()
    return round(prediction)