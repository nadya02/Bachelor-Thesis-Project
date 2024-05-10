import wget, os, gzip, pickle, random, re, sys
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from random import shuffle


#Constants
IMDB_URL = 'http://dlvu.github.io/data/imdb.{}.pkl.gz'
IMDB_FILE = 'imdb.{}.pkl.gz'
PAD, START, END, UNK = '.pad', '.start', '.end', '.unk'

#Configurations
learning_rates = [0.001]
optimizers = ['Adam']
embedded_dim = 300
hidden_dim = 300
num_epochs = 2
batch_size = 128
print_interval = 20
max_length = 1000
max_batch_tokens = 5000

def load_imdb(final=False, val=5000, seed=0, voc=None, char=False):

    cst = 'char' if char else 'word'

    imdb_url = IMDB_URL.format(cst)
    imdb_file = IMDB_FILE.format(cst)

    if not os.path.exists(imdb_file):
        wget.download(imdb_url)

    with gzip.open(imdb_file) as file:
        sequences, labels, i2w, w2i = pickle.load(file)

    if voc is not None and voc < len(i2w):
        nw_sequences = {}

        i2w = i2w[:voc]
        w2i = {w: i for i, w in enumerate(i2w)}

        mx, unk = voc, w2i['.unk']
        for key, seqs in sequences.items():
            nw_sequences[key] = []
            for seq in seqs:
                seq = [s if s < mx else unk for s in seq]
                nw_sequences[key].append(seq)

        sequences = nw_sequences

    if final:
        return (sequences['train'], labels['train']), (sequences['test'], labels['test']), (i2w, w2i), 2

    # Make a validation split
    random.seed(seed)

    x_train, y_train = [], []
    x_val, y_val = [], []

    val_ind = set( random.sample(range(len(sequences['train'])), k=val) )
    for i, (s, l) in enumerate(zip(sequences['train'], labels['train'])):
        if i in val_ind:
            x_val.append(s)
            y_val.append(l)
        else:
            x_train.append(s)
            y_train.append(l)

    return (x_train, y_train), \
           (x_val, y_val), \
           (i2w, w2i), 2

def create_fixed_batches(reviews, labels, batch_size, max_seq_length, pad_index):
    batches = []
    for i in range(0, len(reviews), batch_size):
        batch_reviews = reviews[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        padded_reviews = [review[:max_seq_length] + [pad_index] * max(0, max_seq_length - len(review)) for review in batch_reviews]
        batches.append((padded_reviews, batch_labels))
    return batches

# def add_padding(curr_sequences, pad_index):
#     # change to the get max length functioin?
#     max_length = len(curr_sequences[-1])
#     padded_sequences = []
#     for i in range(len(curr_sequences)):
#         length_to_pad = max_length - len(curr_sequences[i])
#         padded_sequences.append(curr_sequences[i] + [pad_index] * length_to_pad)
#     return padded_sequences

# def split_into_batches(reviews_labeled, batch_size, pad_index):
#     batches_padded_labeled = []
#     for idx in range(0, len(reviews_labeled), batch_size):
#         # print(f"Index {idx}: ")
#         curr_batch = reviews_labeled[idx:(idx + batch_size)]
#         curr_reviews, curr_labels = zip(*curr_batch)
#         batches_padded_labeled.append((add_padding(curr_reviews, pad_index), curr_labels))
#     return batches_padded_labeled

# def create_tensors(reviews, labels, batch_size, pad_index):
    
#     reviews_and_labels = list(zip(reviews, labels))
#     batches_labeled = split_into_batches(reviews_and_labels, batch_size, pad_index)
    
#     shuffle(batches_labeled)
#     tensor_batches = [
#         (torch.tensor(reviews_batch, dtype=torch.long), torch.tensor(labels_batch, dtype=torch.long))
#         for reviews_batch, labels_batch in batches_labeled
#         ]
#     return tensor_batches

def dynamic_batching_2(reviews, labels, max_length, max_batch_tokens, pad_index):
    proccessed_batches = []
    curr_batch = []
    token_count = 0
    # num_batches = -1
    longest_review = 0
    for review, label in zip(reversed(reviews), reversed(labels)):
        if token_count == 0:
            longest_review = len(review)
            # print(f"Longest review: {longest_review}")
            # print(f"Numb_batches: {max_batch_tokens / longest_review}")
            
        if len(review) > longest_review:  
            processed_review = review[:max_length]
        else:
            length_to_pad = longest_review - len(review)
            processed_review = review + [pad_index] * length_to_pad
        
        curr_batch.append((processed_review, label))
        token_count += longest_review
        
        if token_count > max_batch_tokens:
            proccessed_batches.append(curr_batch)
            curr_batch = []
            token_count = 0
     
    if curr_batch:
        proccessed_batches.append(curr_batch)
    
    return proccessed_batches

# def create_tensors(reviews, labels, max_length, max_batch_tokens, pad_index):
    
#     # reviews_and_labels = list(zip(reviews, labels))
#     batches_labeled = dynamic_batching_2(reviews, labels, max_length, max_batch_tokens, pad_index)
    
#     shuffle(batches_labeled)
#     tensor_batches = [
#         (torch.tensor([review for review, _ in batch], dtype=torch.long), 
#          torch.tensor([label for _, label in batch], dtype=torch.long))
#         for batch in batches_labeled
#     ]
#     return tensor_batches

def create_tensors(reviews, labels, batch_size, max_seq_length, pad_index):
    batches_labeled = create_fixed_batches(reviews, labels, batch_size, max_seq_length, pad_index)
    
    shuffle(batches_labeled)
    tensor_batches = [
        (torch.tensor(padded_reviews, dtype=torch.long), torch.tensor(batch_labels, dtype=torch.long))
        for padded_reviews, batch_labels in batches_labeled
    ]
    return tensor_batches

def compute_metrics(model, tensor_batches, loss_func):
    model.eval()
    correct_pred, num_examples, total_loss = 0, 0, 0
    with torch.no_grad():
        for reviews, labels in tensor_batches:
            outputs = model(reviews)
            loss = loss_func(outputs, labels)
            
            total_loss += loss.item() * reviews.size(0)
            _, predicted_labels = torch.max(outputs, 1)

            num_examples += labels.size(0)
            correct_pred += (predicted_labels == labels).sum()
        avg_loss =  total_loss / num_examples
        accuracy = correct_pred.float()/num_examples * 100
    return avg_loss, accuracy

class SimpleSelfAtention(nn.Module):
    def __init__(self, vocab_size, embedd_dim, hidden_dim, numcls):
        super().__init__()
        self.embedd_dim = embedd_dim
        self.embedding = nn.Embedding(vocab_size, embedd_dim)  
        
        self.w_q = nn.Linear(embedd_dim, embedd_dim, bias = False)
        self.w_k = nn.Linear(embedd_dim, embedd_dim, bias = False)
        self.w_v = nn.Linear(embedd_dim, embedd_dim, bias = False)
        
        self.linear1 = nn.Linear(embedd_dim, hidden_dim)
        self.activation1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        
        self.classifier = nn.Linear(hidden_dim, numcls)
        
    @staticmethod    
    def attention(query, key, value):
        embedded_dim = query.shape[-1]
        #(batch, seq_length, embed_dim) ---> (batch, seq_length, seq_length)
        attention_scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(embedded_dim)
    
        attention_scores = F.softmax(attention_scores, dim=-1)
        
        return torch.bmm(attention_scores, value), attention_scores
    
        
    def forward(self, x):
        # (b, s) ---> (b, s, e) 
        # print(f"Embedding input shape: {x.shape}")
        x = self.embedding(x) 
        batch_size, sequence_length, embedded_dim = x.size()
        
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        attention_output, self.attention_scores = SimpleSelfAtention.attention(Q, K, V)
        ff_output = self.activation1(self.linear1(attention_output))
        ff_output, _ = torch.max(ff_output, 1)
        ff_output = self.drop1(ff_output)

        ff_output = self.classifier(ff_output)
        return ff_output
    

def get_optimizer(optimizer_name, model_parameters, lr):
    if optimizer_name == 'SGD':
        return optim.SGD(model_parameters, lr=lr, momentum=0.9)
    elif optimizer_name == 'Adam':
        return optim.Adam(model_parameters, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
def main():
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)
    pad_index = w2i[PAD]
    print(len(x_val))
    vocab_size = len(i2w)

    results = []

    for optimizer_name in optimizers:
        for lr in learning_rates:
            print(f"Optimizer: {optimizer_name} with {lr} learning rate")
            model = SimpleSelfAtention(vocab_size, embedded_dim, hidden_dim, numcls)
            loss_func = nn.CrossEntropyLoss()
            optimizer = get_optimizer(optimizer_name, model.parameters(), lr)
            writer = SummaryWriter(log_dir='logs/simple_experiment')
            
            train_losses = []
            val_accuracies = []

            # tensor_batches_train = create_tensors(x_train, y_train, batch_size, pad_index)
            # tensor_batches_val = create_tensors(x_val, y_val, batch_size, pad_index)
            
            tensor_batches_train = create_tensors(x_train, y_train, max_length, max_batch_tokens, pad_index)
            tensor_batches_val = create_tensors(x_val, y_val, max_length, max_batch_tokens, pad_index)
            
            for epoch in range(num_epochs):
                model.train()
                for i, (reviews, labels) in enumerate(tensor_batches_train):
                    outputs = model(reviews)
                    loss = loss_func(outputs, labels)
                    
                    # backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    
                    if (i + 1) % print_interval == 0:
                        print(
                            f"Epoch [{epoch + 1} / {num_epochs}], "
                            f"Step [{i + 1}/{len(tensor_batches_train)}], "
                            f"Loss{loss.item(): .4f}"
                        )
                        
                train_loss, train_accuracy = compute_metrics(model, tensor_batches_train, loss_func)
                val_loss, val_accuracy = compute_metrics(model, tensor_batches_val, loss_func)
                val_accuracies.append(val_accuracy)
                train_losses.append(train_loss)
                
                print(
                    f"Training Loss: {train_loss: .4f}, Training Accuracy: {train_accuracy:.2f}%"
                    f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
                )
            results.append((optimizer_name, lr, train_losses, val_accuracies))

if __name__ == "__main__":
    main()