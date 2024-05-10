import wget, os, gzip, pickle, random, re, sys
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import statistics
from random import shuffle
from datetime import datetime


#Constants
IMDB_URL = 'http://dlvu.github.io/data/imdb.{}.pkl.gz'
IMDB_FILE = 'imdb.{}.pkl.gz'
PAD, START, END, UNK = '.pad', '.start', '.end', '.unk'

#Configurations
learning_rates = [0.0001]
optimizers = ['Adam']
embed_dim = 216
d_ff = 512 # Should it be double??
num_epochs = 2
# Example usage
max_seq_len = 774
# max_batch_tokens = 3000
batch_size = 32
print_interval = 10
heads = 4
N = 1

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


# def dynamic_batching_2(reviews, labels, max_length, max_batch_tokens, pad_index):
#     proccessed_batches = []
#     curr_batch = []
#     token_count = 0
#     # num_batches = -1
#     longest_review = 0
#     for review, label in zip(reversed(reviews), reversed(labels)):
#         if token_count == 0:
#             longest_review = len(review)
#             # print(f"Longest review: {longest_review}")
#             # print(f"Numb_batches: {max_batch_tokens / longest_review}")
            
#         if len(review) > longest_review:  
#             processed_review = review[:max_length]
#         else:
#             length_to_pad = longest_review - len(review)
#             processed_review = review + [pad_index] * length_to_pad
        
#         curr_batch.append((processed_review, label))
#         token_count += longest_review
        
#         if token_count > max_batch_tokens:
#             proccessed_batches.append(curr_batch)
#             curr_batch = []
#             token_count = 0
     
#     if curr_batch:
#         proccessed_batches.append(curr_batch)
    
#     return proccessed_batches

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


def truncate_and_padding(curr_sequences, max_seq_len, pad_index):
    # change to the get max length functioin?
    max_length = len(curr_sequences[-1])
    padded_sequences = []
    for i in range(len(curr_sequences)):
        curr_sequences = curr_sequences[:max_seq_len + 1]
        length_to_pad = max_length - len(curr_sequences[i])
        padded_sequences.append(curr_sequences[i] + [pad_index] * length_to_pad)
    return padded_sequences

def split_into_batches(reviews_labeled, batch_size, max_seq_len, pad_index):
    batches_padded_labeled = []
    for idx in range(0, len(reviews_labeled), batch_size):
        # print(f"Index {idx}: ")
        curr_batch = reviews_labeled[idx:(idx + batch_size)]
        curr_reviews, curr_labels = zip(*curr_batch)
        batches_padded_labeled.append((truncate_and_padding(curr_reviews, max_seq_len, pad_index), curr_labels))
    return batches_padded_labeled

def create_tensors(reviews, labels, batch_size, max_seq_len, pad_index):
    
    reviews_and_labels = list(zip(reviews, labels))
    batches_labeled = split_into_batches(reviews_and_labels, batch_size, max_seq_len, pad_index)
    
    shuffle(batches_labeled)
    tensor_batches = [
        (torch.tensor(reviews_batch, dtype=torch.long), torch.tensor(labels_batch, dtype=torch.long))
        for reviews_batch, labels_batch in batches_labeled
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

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)  
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

    
    def forward(self, x):
        # (batch, seq_length) ---> (batch, seq_length, embed_size) 
        
        batch_size, seq_length = x.size()
        token_embed = self.embedding(x)
        
        #positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)
        positions = torch.arange(seq_length)
        if positions.max() >= self.pos_embedding.num_embeddings:
            # print("Max position index:", positions.max().item())
            # print("Max token index:", x.max().item())
            raise ValueError("Position index out of range: Max position should be less than {}".format(self.pos_embedding.num_embeddings))
        pos_embed = self.pos_embedding(positions)[None, :, :].expand(batch_size, seq_length, self.embed_dim)
        
        return token_embed + pos_embed

        # return token_embed
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, heads = 4):
        super().__init__()

        assert embed_dim % heads == 0
        
        self.embed_dim = embed_dim
        self.heads = heads
        self.d_k = embed_dim // heads # Dimenstions of vector seen by each head
        
        self.w_q = nn.Linear(embed_dim, embed_dim, bias = False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias = False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias = False)
        
        self.w_o = nn.Linear(embed_dim, embed_dim, bias = False)
        
        
    @staticmethod    
    def attention(query, key, value):
        d_k = query.shape[-1]
        
        #(batch, head, seq_len, d_k) ---> (batch, head, seq_length, seq_length)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        
        #print(f"Q: {query.shape} x K.T{key.transpose(1, 2).shape} = {attention_scores.shape}")  
        attention_scores = F.softmax(attention_scores, dim=-1)
        
        #(batch, head, seq_length, seq_length) --> (batch, head, seq_len, d_k)
        return torch.matmul(attention_scores, value)
        
    def forward(self, x):
        # print(f"Multihead input shape: {x.shape}")
        batch_size, sequence_length, _ = x.size()
        
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        # print(f"Shapes Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
        
        # Split each tensor into heads, where each head has size d_k
        
        # (batch, seq_length, embed_dim) --> (batch, seq_len, head, embed_dim) --> (batch, head, seq_len, embed_dim)
        Q = Q.view(batch_size, sequence_length, self.heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, sequence_length, self.heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, sequence_length, self.heads, self.d_k).transpose(1, 2)
        # print(f"Shapes Q': {Q.shape}, K': {K.shape}, V': {V.shape}")
        
        attention_output = MultiHeadSelfAttention.attention(Q, K, V)
        
        # (batch, head, seq_len, d_k) --> (batch, seq_len, head, d_k) --> (batch, seq_len, head * d_k)
        combined_heads = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.heads * self.d_k)
        # print(f"Multi-Head Attention output shape: {combined_heads.shape}")
        return self.w_o(combined_heads)
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, d_ff):          
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, heads)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, embed_dim)
        )
        
    def forward(self, x):
        attention = self.attention(x)
        x = self.norm1(attention + x)
        
        fforward = self.feed_forward(x)
        return self.norm2(fforward + x)
        
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, heads, d_ff, numcls, max_seq_len, N):
        super().__init__()
        
        self.embedding_layer = InputEmbedding(vocab_size, embed_dim, max_seq_len)
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(embed_dim, heads, d_ff) for i in range(N)]
        )
        
        # self.transformer_block = TransformerBlock(embed_dim, heads, d_ff)
        
        self.classifier = nn.Linear(embed_dim, numcls)
    
    def forward(self, x):
        x = self.embedding_layer(x)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        x = x.mean(dim=1)
        # x = self.drop1(x)

        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
    
    

def get_optimizer(optimizer_name, model_parameters, lr):
    if optimizer_name == 'SGD':
        return optim.SGD(model_parameters, lr=lr, momentum=0.9)
    elif optimizer_name == 'Adam':
        return optim.Adam(model_parameters, lr=lr, eps=1e-9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
def main():
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)
    pad_index = w2i[PAD]
    print(len(x_val))
    vocab_size = len(i2w)
    print(f"Vocabulary size: {vocab_size}")
    
    lengths = [len(seq) for seq in x_train]
    
    print("Mean: ", statistics.mean(lengths))
    print("Standard deviation: ", statistics.stdev(lengths))
    print("Maximum: ", max(lengths))
    print("Minimum: ", min(lengths))
    
    set_seed(42)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device ", device)

    results = []

    for optimizer_name in optimizers:
        for lr in learning_rates:
            print(f"Optimizer: {optimizer_name} with {lr} learning rate")
            model = TransformerModel(vocab_size, embed_dim, heads, d_ff, numcls, max_seq_len, N).to(device)
            loss_func = nn.CrossEntropyLoss()
            optimizer = get_optimizer(optimizer_name, model.parameters(), lr)
            
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            log_dir = f'logs/experiment_{timestamp}'
            writer = SummaryWriter(log_dir=log_dir)
            
            train_losses = []
            val_accuracies = []

            tensor_batches_train = create_tensors(x_train, y_train, batch_size, max_seq_len, pad_index)
            tensor_batches_val = create_tensors(x_val, y_val, batch_size, max_seq_len, pad_index)
            
            for epoch in range(num_epochs):
                model.train()
                for i, (reviews, labels) in enumerate(tensor_batches_train):
                    reviews, labels = reviews.to(device), labels.to(device)
                    outputs = model(reviews)
                    loss = loss_func(outputs, labels)
                    
                    # backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    
                    if (i + 1) % print_interval == 0:
                        print(
                            f"Epoch [{epoch + 1} / {num_epochs}], "
                            f"Step [{i + 1}/{len(tensor_batches_train)}], "
                            f"Loss{loss.item(): .4f}"
                        )
                        
                    writer.add_scalar('Training Loss', loss.item(), epoch * len(tensor_batches_train) + i)
                        
                train_loss, train_accuracy = compute_metrics(model, tensor_batches_train, loss_func)
                val_loss, val_accuracy = compute_metrics(model, tensor_batches_val, loss_func)
                writer.add_scalar('Validation Loss', val_loss, epoch)
                
                val_accuracies.append(val_accuracy)
                train_losses.append(train_loss)
                
                
                print(
                    f"Training Loss: {train_loss: .4f}, Training Accuracy: {train_accuracy:.2f}%"
                    f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
                )
            results.append((optimizer_name, lr, train_losses, val_accuracies))
            writer.close()

if __name__ == "__main__":
    main()