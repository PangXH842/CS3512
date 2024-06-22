import time
import pandas as pd
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Training function with mixed precision
def train_epoch(model, data_loader, optimizer, device, scaler):
    model.train()
    total_loss = 0
    for inputs, masks, labels in tqdm(data_loader, desc="Training", leave=False):
        inputs = inputs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, masks, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs, attention_mask=masks)
            predictions.extend(outputs.logits.argmax(dim=1).tolist())
            true_labels.extend(labels.tolist())
    return accuracy_score(true_labels, predictions)

def main(args):
    # Check if GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir, num_labels=2).to(device)

    # Load dataset
    if not os.path.isfile(args.dataset_dir):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_dir}")
    
    print(f"Loading dataset from {args.dataset_dir}...")
    start_time = time.time()
    df = pd.read_csv(args.dataset_dir)
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds.")

    # Limit the dataset for quick experimentation
    df = df.sample(n=min(5000, len(df)), random_state=42)  

    # Preprocess the dataset
    texts = df['review'].tolist()
    label_mapping = {'positive': 1, 'negative': 0}
    labels = [label_mapping[label] for label in df['sentiment'].tolist()]
    labels = torch.tensor(labels)

    # Tokenize the texts
    input_ids = []
    attention_masks = []

    for text in tqdm(texts, desc="Tokenizing"):
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,  # Ensure long texts are truncated
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    # Convert lists to tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Split the data into training and validation sets
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, random_state=2024, test_size=0.1)
    train_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2024, test_size=0.1)

    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler()  # For mixed precision training

    # Start timing
    overall_start_time = time.time()

    # Training and evaluation loop
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{args.epochs} started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch_start_time))}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        
        epoch_end_time = time.time()
        val_acc = evaluate(model, val_loader, device)
        print(f'Epoch {epoch+1} finished at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_end_time))}')
        print(f'Time taken for Epoch {epoch+1}: {epoch_end_time - epoch_start_time:.2f} seconds')
        print(f'Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}')

    overall_end_time = time.time()
    print(f"\nTotal time consumed (s): {overall_end_time - overall_start_time:.2f}")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="bert_base_uncased")
    parser.add_argument('--dataset_dir', type=str, default="IMDB.csv")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    
    args = parser.parse_args()
    main(args)
