#!/usr/bin/env python3
"""
English-French Neural Machine Translation System

This module implements a Transformer-based neural machine translation system
for translating English text to French. It includes training, evaluation,
and inference capabilities with document summarization.

Author: CS6120 Project
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import pickle
import os
import logging
import re
import random
import unicodedata
import csv
from typing import Dict, List, Tuple, Optional, Any
from random import shuffle
from collections import Counter
from sklearn.model_selection import KFold
import sacrebleu
from rouge_score import rouge_scorer

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Special tokens
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3


class MultiHeadAttentionLayer(nn.Module):
    """Multi-Head Attention mechanism for Transformer architecture."""
    
    def __init__(
        self, hidden_size: int, n_heads: int, dropout: float, device: torch.device
    ) -> None:
        super().__init__()
        
        assert (
            hidden_size % n_heads == 0
        ), "Hidden size must be divisible by the number of heads."
        
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_size = hidden_size // n_heads
        
        # Linear layers for query, key, and value projections
        self.fc_query = nn.Linear(hidden_size, hidden_size)
        self.fc_key = nn.Linear(hidden_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        
        self.dp = nn.Dropout(dropout)
        self.coefficient = torch.sqrt(torch.FloatTensor([self.head_size])).to(device)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b_size = query.shape[0]
        
        # Linear projections
        query_output = self.fc_query(query)
        key_output = self.fc_key(key)
        value_output = self.fc_value(value)
        
        # Reshape and permute for multi-head attention
        query_output = query_output.view(
            b_size, -1, self.n_heads, self.head_size
        ).permute(0, 2, 1, 3)
        key_output = key_output.view(b_size, -1, self.n_heads, self.head_size).permute(
            0, 2, 1, 3
        )
        value_output = value_output.view(
            b_size, -1, self.n_heads, self.head_size
        ).permute(0, 2, 1, 3)
        
        # Calculate attention scores
        energy = (
            torch.matmul(query_output, key_output.permute(0, 1, 3, 2))
            / self.coefficient
        )
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # Apply softmax to get attention weights
        attention = torch.softmax(energy, dim=-1)
        
        # Calculate the weighted sum of values
        output = torch.matmul(self.dp(attention), value_output)
        
        # Concatenate heads and pass through the final linear layer
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(b_size, -1, self.hidden_size)
        output = self.fc_out(output)
        
        return output, attention


class FeedForwardLayer(nn.Module):
    """Feed-forward neural network layer for Transformer."""
    
    def __init__(self, hidden_size: int, ff_size: int, dropout: float) -> None:
        super().__init__()
        
        self.ff_layer = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.ff_layer(input)


class EncoderLayer(nn.Module):
    """Single layer of the Transformer encoder."""
    
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        ff_size: int,
        dropout: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        
        self.self_atten = MultiHeadAttentionLayer(hidden_size, n_heads, dropout, device)
        self.self_atten_norm = nn.LayerNorm(hidden_size)
        self.ff_layer = FeedForwardLayer(hidden_size, ff_size, dropout)
        self.dp = nn.Dropout(dropout)
        self.ff_layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, input: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention
        atten_result, _ = self.self_atten(input, input, input, input_mask)
        
        # Add & norm
        atten_norm = self.self_atten_norm(input + self.dp(atten_result))
        
        # Feed-forward
        ff_result = self.ff_layer(atten_norm)
        
        # Add & norm
        output = self.ff_layer_norm(atten_norm + self.dp(ff_result))
        
        return output


class Encoder(nn.Module):
    """Complete Transformer encoder with multiple layers."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int,
        n_heads: int,
        ff_size: int,
        dropout: float,
        device: torch.device,
        MAX_LENGTH: int = 100,
    ) -> None:
        super().__init__()
        
        self.device = device
        
        # Embedding layers for tokens and positions
        self.te = nn.Embedding(input_size, hidden_size)
        self.pe = nn.Embedding(MAX_LENGTH, hidden_size)
        
        # Stack of encoder layers
        encoding_layers = [
            EncoderLayer(hidden_size, n_heads, ff_size, dropout, device)
            for _ in range(n_layers)
        ]
        self.encode_sequence = nn.Sequential(*encoding_layers)
        
        self.dp = nn.Dropout(dropout)
        self.coefficient = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)
    
    def forward(self, input: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        b_size, input_size = input.shape
        
        # Create position tensor and add positional embeddings
        pos = torch.arange(0, input_size).unsqueeze(0).repeat(b_size, 1).to(self.device)
        input = self.dp((self.te(input) * self.coefficient) + self.pe(pos))
        
        # Pass through each encoder layer
        for layer in self.encode_sequence:
            input = layer(input, input_mask)
        
        return input


class DecoderLayer(nn.Module):
    """Single layer of the Transformer decoder."""
    
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        ff_size: int,
        dropout: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        
        self.self_atten = MultiHeadAttentionLayer(hidden_size, n_heads, dropout, device)
        self.self_atten_norm = nn.LayerNorm(hidden_size)
        self.encoder_atten = MultiHeadAttentionLayer(
            hidden_size, n_heads, dropout, device
        )
        self.encoder_atten_norm = nn.LayerNorm(hidden_size)
        self.ff_layer = FeedForwardLayer(hidden_size, ff_size, dropout)
        self.ff_layer_norm = nn.LayerNorm(hidden_size)
        self.dp = nn.Dropout(dropout)
    
    def forward(
        self,
        target: torch.Tensor,
        encoded_input: torch.Tensor,
        target_mask: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        atten_result, _ = self.self_atten(target, target, target, target_mask)
        atten_norm = self.self_atten_norm(target + self.dp(atten_result))
        
        # Encoder-decoder attention
        atten_encoded, attention = self.encoder_atten(
            atten_norm, encoded_input, encoded_input, input_mask
        )
        encoded_norm = self.encoder_atten_norm(atten_norm + self.dp(atten_encoded))
        
        # Feed-forward
        ff_result = self.ff_layer(encoded_norm)
        output = self.ff_layer_norm(encoded_norm + self.dp(ff_result))
        
        return output, attention


class Decoder(nn.Module):
    """Complete Transformer decoder with multiple layers."""
    
    def __init__(
        self,
        output_size: int,
        hidden_size: int,
        n_layers: int,
        n_heads: int,
        ff_size: int,
        dropout: float,
        device: torch.device,
        MAX_LENGTH: int = 100,
    ) -> None:
        super().__init__()
        
        self.device = device
        
        # Embedding layers for tokens and positions
        self.te = nn.Embedding(output_size, hidden_size)
        self.pe = nn.Embedding(MAX_LENGTH, hidden_size)
        
        # Stack of decoder layers
        decoding_layers = [
            DecoderLayer(hidden_size, n_heads, ff_size, dropout, device)
            for _ in range(n_layers)
        ]
        self.decode_sequence = nn.Sequential(*decoding_layers)
        
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dp = nn.Dropout(dropout)
        self.coefficient = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)
    
    def forward(
        self,
        target: torch.Tensor,
        encoded_input: torch.Tensor,
        target_mask: torch.Tensor,
        input_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b_size, target_size = target.shape
        
        # Create position tensor and add positional embeddings
        pos = (
            torch.arange(0, target_size).unsqueeze(0).repeat(b_size, 1).to(self.device)
        )
        target = self.dp((self.te(target) * self.coefficient) + self.pe(pos))
        
        # Pass through each decoder layer
        for layer in self.decode_sequence:
            target, attention = layer(target, encoded_input, target_mask, input_mask)
        
        # Final linear layer to generate output predictions
        output = self.fc_out(target)
        
        return output, attention


class Transformer(nn.Module):
    """Complete Transformer model combining encoder and decoder."""
    
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        device: torch.device,
        padding_index: int = 0,
    ) -> None:
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.padding_index = padding_index
        self.device = device
    
    def make_input_mask(self, input: torch.Tensor) -> torch.Tensor:
        """Create input mask to ignore padding tokens."""
        input_mask = (input != self.padding_index).unsqueeze(1).unsqueeze(2)
        return input_mask
    
    def make_target_mask(self, target: torch.Tensor) -> torch.Tensor:
        """Create target mask for autoregressive generation."""
        target_pad_mask = (target != self.padding_index).unsqueeze(1).unsqueeze(2)
        target_sub_mask = torch.tril(
            torch.ones((target.shape[1], target.shape[1]), device=self.device)
        ).bool()
        target_mask = target_pad_mask & target_sub_mask
        return target_mask
    
    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_mask = self.make_input_mask(input)
        target_mask = self.make_target_mask(target)
        
        # Encode input sequences
        encoded_input = self.encoder(input, input_mask)
        
        # Decode target sequences with encoded input
        output, attention = self.decoder(target, encoded_input, target_mask, input_mask)
        
        return output, attention


class Dictionary:
    """Vocabulary management for the translation model."""
    
    def __init__(self, name: str) -> None:
        self.name = name
        self.word2index: Dict[str, int] = {
            "<pad>": PAD_TOKEN,
            "<sos>": SOS_TOKEN,
            "<eos>": EOS_TOKEN,
            "<unk>": UNK_TOKEN,
        }
        self.word2count: Dict[str, int] = {}
        self.index2word: Dict[int, str] = {
            PAD_TOKEN: "<pad>",
            SOS_TOKEN: "<sos>",
            EOS_TOKEN: "<eos>",
            UNK_TOKEN: "<unk>",
        }
        self.n_count: int = 4  # Count includes PAD, SOS, EOS, and UNK
    
    def add_sentence(self, sentence: str) -> None:
        """Add all words in a sentence to the dictionary."""
        for word in sentence.split(" "):
            self.add_word(word)
    
    def add_word(self, word: str) -> None:
        """Add a word to the dictionary."""
        if word not in self.word2index:
            self.word2index[word] = self.n_count
            self.word2count[word] = 1
            self.index2word[self.n_count] = word
            self.n_count += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s: str) -> str:
    """Convert Unicode to ASCII."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalizeString(s: str) -> str:
    """Normalize string for processing."""
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def load_files(
    lang1: str,
    lang2: str,
    data_dir: str,
    reverse: bool = True,
    MAX_FILE_SIZE: int = 100000,
    MAX_LENGTH: int = 60,
) -> Tuple[Dictionary, Dictionary, List[str], List[str]]:
    """Load and preprocess language files."""
    lang1_list = []
    lang2_list = []
    
    # Find CSV file in data directory
    csv_file_path = None
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            if file_name.endswith(".csv"):
                csv_file_path = os.path.join(root, file_name)
                break
        if csv_file_path:
            break
    
    if not csv_file_path:
        raise FileNotFoundError(f"CSV file not found in {data_dir}")
    
    # Read CSV data
    all_lang1_lines = []
    all_lang2_lines = []
    
    with open(csv_file_path, mode="r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            all_lang1_lines.append(row[lang1.capitalize()].strip())
            all_lang2_lines.append(row[lang2.capitalize()].strip())
    
    # Ensure both files have the same number of lines
    assert len(all_lang1_lines) == len(
        all_lang2_lines
    ), "Mismatched number of lines in language files"
    
    # Sample data based on MAX_FILE_SIZE
    interval = max(1, len(all_lang1_lines) // MAX_FILE_SIZE)
    lang1_list = [all_lang1_lines[i] for i in range(0, len(all_lang1_lines), interval)]
    lang2_list = [all_lang2_lines[i] for i in range(0, len(all_lang2_lines), interval)]
    
    # Limit to MAX_FILE_SIZE
    lang1_list = lang1_list[:MAX_FILE_SIZE]
    lang2_list = lang2_list[:MAX_FILE_SIZE]
    
    logging.info(f"Loaded {len(lang1_list)} sentences for {lang1}")
    logging.info(f"Loaded {len(lang2_list)} sentences for {lang2}")
    
    # Preprocess strings
    lang1_normalized = list(map(normalizeString, lang1_list))
    lang2_normalized = list(map(normalizeString, lang2_list))
    
    # Filter by length
    lang1_sentences = []
    lang2_sentences = []
    
    for i in range(len(lang1_normalized)):
        tokens1 = lang1_normalized[i].split(" ")
        tokens2 = lang2_normalized[i].split(" ")
        if len(tokens1) <= MAX_LENGTH and len(tokens2) <= MAX_LENGTH:
            lang1_sentences.append(lang1_normalized[i])
            lang2_sentences.append(lang2_normalized[i])
    
    logging.info(f"{len(lang1_sentences)} {lang1} sentences after length filtering")
    logging.info(f"{len(lang2_sentences)} {lang2} sentences after length filtering")
    
    if reverse:
        input_dic = Dictionary(lang2)
        output_dic = Dictionary(lang1)
        return input_dic, output_dic, lang2_sentences, lang1_sentences
    else:
        input_dic = Dictionary(lang1)
        output_dic = Dictionary(lang2)
        return input_dic, output_dic, lang1_sentences, lang2_sentences


def tokenize(sentence: str, dictionary: Dictionary, MAX_LENGTH: int = 60) -> List[int]:
    """Tokenize sentence using dictionary."""
    split_sentence = [word for word in sentence.split(" ")]
    token = [SOS_TOKEN]
    token += [
        dictionary.word2index.get(word, dictionary.word2index["<unk>"])
        for word in sentence.split(" ")
    ]
    token.append(EOS_TOKEN)
    token += [PAD_TOKEN] * (MAX_LENGTH - len(split_sentence))
    return token


def load_batches(
    input_lang: List[List[int]],
    output_lang: List[List[int]],
    batch_size: int,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create batches for training."""
    data_loader = []
    for i in range(0, len(input_lang), batch_size):
        input_batch = input_lang[i : i + batch_size]
        target_batch = output_lang[i : i + batch_size]
        
        if len(input_batch) == 0 or len(target_batch) == 0:
            continue
        
        input_tensor = torch.LongTensor(input_batch).to(device)
        target_tensor = torch.LongTensor(target_batch).to(device)
        data_loader.append([input_tensor, target_tensor])
    return data_loader


class Trainer:
    """Training class for the Transformer model."""
    
    def __init__(
        self,
        lang1: str,
        lang2: str,
        data_directory: str,
        reverse: bool,
        MAX_LENGTH: int,
        MAX_FILE_SIZE: int,
        batch_size: int,
        lr: float = 0.0005,
        hidden_size: int = 256,
        encoder_layers: int = 3,
        decoder_layers: int = 3,
        encoder_heads: int = 8,
        decoder_heads: int = 8,
        encoder_ff_size: int = 512,
        decoder_ff_size: int = 512,
        encoder_dropout: float = 0.1,
        decoder_dropout: float = 0.1,
        device: str = "cpu",
    ) -> None:
        """Initialize the Trainer."""
        self.MAX_LENGTH = MAX_LENGTH
        self.MAX_FILE_SIZE = MAX_FILE_SIZE
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_size = hidden_size
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_heads = encoder_heads
        self.decoder_heads = decoder_heads
        self.encoder_ff_size = encoder_ff_size
        self.decoder_ff_size = decoder_ff_size
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        
        # Load language data and create dictionaries
        (
            self.input_lang_dic,
            self.output_lang_dic,
            self.input_lang_list,
            self.output_lang_list,
        ) = load_files(
            lang1, lang2, data_directory, reverse, self.MAX_FILE_SIZE, self.MAX_LENGTH
        )
        
        if self.input_lang_dic is None or self.output_lang_dic is None:
            raise ValueError("Loading language files failed due to mismatched line counts.")
        
        # Add sentences to dictionaries
        for sentence in self.input_lang_list:
            self.input_lang_dic.add_sentence(sentence)
        for sentence in self.output_lang_list:
            self.output_lang_dic.add_sentence(sentence)
        
        # Save dictionaries
        self.save_dictionary(self.input_lang_dic, input=True)
        self.save_dictionary(self.output_lang_dic, input=False)
        
        # Tokenize sentences
        self.tokenized_input_lang = [
            tokenize(sentence, self.input_lang_dic, self.MAX_LENGTH)
            for sentence in self.input_lang_list
        ]
        self.tokenized_output_lang = [
            tokenize(sentence, self.output_lang_dic, self.MAX_LENGTH)
            for sentence in self.output_lang_list
        ]
        
        # Create data loader
        self.data_loader = load_batches(
            self.tokenized_input_lang,
            self.tokenized_output_lang,
            self.batch_size,
            self.device,
        )
        
        # Define model sizes
        input_size = self.input_lang_dic.n_count
        output_size = self.output_lang_dic.n_count
        
        logging.info(f"Input vocabulary size: {input_size}")
        logging.info(f"Output vocabulary size: {output_size}")
        
        # Create model components
        encoder_part = Encoder(
            input_size,
            hidden_size,
            encoder_layers,
            encoder_heads,
            encoder_ff_size,
            encoder_dropout,
            self.device,
        )
        decoder_part = Decoder(
            output_size,
            hidden_size,
            decoder_layers,
            decoder_heads,
            decoder_ff_size,
            decoder_dropout,
            self.device,
        )
        
        # Initialize transformer
        self.transformer = Transformer(
            encoder_part, decoder_part, self.device, PAD_TOKEN
        ).to(self.device)
        self.transformer.apply(self.initialize_weights)
        
        # Loss and optimizer
        self.loss_func = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        self.optimizer = optim.Adam(self.transformer.parameters(), lr=lr)
    
    def initialize_weights(self, model: nn.Module) -> None:
        """Initialize model weights using Xavier uniform initialization."""
        if hasattr(model, "weight") and model.weight.dim() > 1:
            nn.init.xavier_uniform_(model.weight.data)
    
    def save_dictionary(self, dictionary: dict, input: bool = True) -> None:
        """Save language dictionary to disk."""
        directory = (
            f"saved_models/{self.input_lang_dic.name}2{self.output_lang_dic.name}"
        )
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        file_path = f"{directory}/{'input_dic.pkl' if input else 'output_dic.pkl'}"
        with open(file_path, "wb") as f:
            pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        shuffle(self.data_loader)
        train_loss = 0
        
        for input, target in self.data_loader:
            if input.size(0) == 0 or target.size(0) == 0:
                logging.warning("Empty batch detected. Skipping...")
                continue
            
            # Ensure tensors have at least 2 dimensions
            if input.dim() == 1:
                input = input.unsqueeze(0)
            if target.dim() == 1:
                target = target.unsqueeze(0)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output, _ = self.transformer(input, target[:, :-1])
            
            # Reshape for loss calculation
            output = output.reshape(-1, output.shape[-1])
            target = target[:, 1:].reshape(-1)
            
            # Calculate loss and backpropagate
            loss = self.loss_func(output, target)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
        
        avg_loss = train_loss / len(self.data_loader)
        perplexity = np.exp(avg_loss)
        return avg_loss, perplexity
    
    def train(self, epochs: int, saved_model_directory: str) -> None:
        """Train the model for specified epochs."""
        for epoch in range(epochs):
            start_time = time.time()
            train_loss, perplexity = self.train_epoch()
            duration = time.time() - start_time
            estimated_remaining_time = (epochs - epoch - 1) * duration
            
            logging.info(
                f"Epoch {epoch + 1}/{epochs}, Time: {duration:.1f}s, "
                f"Estimated remaining time: {estimated_remaining_time:.1f}s"
            )
            logging.info(
                f"  Training Loss: {train_loss:.4f}, Perplexity: {perplexity:.4f}"
            )
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0 or epoch == epochs - 1:
                self.save_model(epoch, saved_model_directory)
        
        logging.info("Training finished!")
        self.final_save_model(saved_model_directory)
    
    def save_model(self, epoch: int, saved_model_directory: str) -> None:
        """Save model checkpoint."""
        directory = os.path.join(
            saved_model_directory,
            f"{self.input_lang_dic.name}2{self.output_lang_dic.name}",
        )
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        model_path = os.path.join(directory, f"transformer_epoch_{epoch}.pth")
        torch.save(self.transformer.state_dict(), model_path)
        logging.info(f"Model saved to {model_path}")
    
    def final_save_model(self, saved_model_directory: str) -> None:
        """Save final model."""
        directory = os.path.join(
            saved_model_directory,
            f"{self.input_lang_dic.name}2{self.output_lang_dic.name}",
        )
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        model_path = os.path.join(directory, "transformer_model.pt")
        torch.save(self.transformer.state_dict(), model_path)
        logging.info(f"Final model saved to {model_path}")


def summarize_document(document: str, max_chars: int = 100) -> str:
    """Summarize document by extracting most frequent sentences."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', document)
    if len(sentences) <= 1:
        return document[:max_chars]
    
    # Count word frequencies
    word_freq = Counter()
    for sentence in sentences:
        words = re.findall(r'\w+', sentence.lower())
        word_freq.update(words)
    
    # Score sentences
    sentence_scores = {}
    for sentence in sentences:
        sentence_scores[sentence] = sum(
            word_freq.get(word.lower(), 0) for word in re.findall(r'\w+', sentence)
        )
    
    # Sort and extract
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary = ' '.join(sorted_sentences)
    
    return summary[:max_chars]


def translate_sentence(
    sentence: str,
    input_dic: Dictionary,
    output_dic: Dictionary,
    model: Transformer,
    device: torch.device,
    max_len: int,
    prob_threshold: float = 0.1,
) -> Tuple[str, Any]:
    """Translate a sentence using the trained model."""
    model.eval()
    normalized_sentence = normalizeString(sentence)
    tokens = tokenize(normalized_sentence, input_dic, max_len)
    
    # Ensure tokens are within valid range
    tokens = [min(token, input_dic.n_count - 1) for token in tokens]
    
    input_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    input_mask = model.make_input_mask(input_tensor)
    
    with torch.no_grad():
        encoded_input = model.encoder(input_tensor, input_mask)
    
    target_tokens = [SOS_TOKEN]
    generated_sentences = set()
    
    for i in range(max_len):
        target_tensor = torch.LongTensor(target_tokens).unsqueeze(0).to(device)
        target_mask = model.make_target_mask(target_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(
                target_tensor, encoded_input, target_mask, input_mask
            )
        
        pred_token = output.argmax(2)[:, -1].item()
        pred_prob = torch.softmax(output, dim=-1)[0, -1, pred_token].item()
        
        # Penalize repetition
        if len(target_tokens) > 1 and pred_token == target_tokens[-1]:
            output[0, -1, pred_token] -= 1.0
            pred_token = output.argmax(2)[:, -1].item()
            pred_prob = torch.softmax(output, dim=-1)[0, -1, pred_token].item()
        
        target_tokens.append(pred_token)
        
        # Stop if probability is too low
        if pred_prob < prob_threshold:
            break
        
        # Check for sentence repetition
        current_sentence = " ".join(
            [output_dic.index2word[t] for t in target_tokens[1:]]
        )
        if current_sentence in generated_sentences:
            output[0, -1, pred_token] -= 1.0
            pred_token = output.argmax(2)[:, -1].item()
            target_tokens[-1] = pred_token
        else:
            generated_sentences.add(current_sentence)
        
        # Stop if EOS token
        if pred_token == EOS_TOKEN:
            break
    
    # Convert tokens to words
    target_results = [output_dic.index2word[i] for i in target_tokens if i != EOS_TOKEN]
    
    return " ".join(target_results[1:]), attention


def translate_documents(
    documents: List[str],
    input_dic: Dictionary,
    output_dic: Dictionary,
    model: Transformer,
    device: torch.device,
    max_len: int,
    prob_threshold: float = 0.1,
) -> List[str]:
    """Translate a list of documents."""
    translated_documents = []
    for document in documents:
        summary = summarize_document(document)
        translation, _ = translate_sentence(
            summary, input_dic, output_dic, model, device, max_len, prob_threshold
        )
        translated_documents.append(translation)
    return translated_documents


def calculate_rouge_scores(hypotheses: List[str], references: List[str]) -> dict:
    """Calculate ROUGE scores for evaluation."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    num_summaries = len(hypotheses)
    
    for hypothesis, reference in zip(hypotheses, references):
        if isinstance(reference, list):
            reference = reference[0]
        if isinstance(hypothesis, list):
            hypothesis = hypothesis[0]
        
        score = scorer.score(reference, hypothesis)
        for key in scores:
            scores[key] += score[key].fmeasure
    
    # Average scores
    scores = {key: value / num_summaries for key, value in scores.items()}
    return scores


def main():
    """Main function to run training and evaluation."""
    # Configuration
    lang1 = 'english'
    lang2 = 'french'
    data_directory = 'data'
    reverse = False
    MAX_LENGTH = 60
    MAX_FILE_SIZE = 200000
    batch_size = 128
    lr = 0.0005
    hidden_size = 256
    encoder_layers = 3
    decoder_layers = 3
    encoder_heads = 8
    decoder_heads = 8
    encoder_ff_size = 512
    decoder_ff_size = 512
    encoder_dropout = 0.1
    decoder_dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f"Using device: {device}")
    
    # Initialize trainer
    trainer = Trainer(
        lang1, lang2, data_directory, reverse, MAX_LENGTH, MAX_FILE_SIZE, batch_size, lr,
        hidden_size, encoder_layers, decoder_layers, encoder_heads, decoder_heads,
        encoder_ff_size, decoder_ff_size, encoder_dropout, decoder_dropout, device
    )
    
    # Train the model
    epochs = 10
    saved_model_directory = './saved_models'
    trainer.train(epochs, saved_model_directory)


if __name__ == "__main__":
    main()
