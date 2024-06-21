import time
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Training function
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, masks, labels in data_loader:
        inputs = inputs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, masks, labels in data_loader:
            inputs = inputs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs, attention_mask=masks)
            predictions.extend(outputs.logits.argmax(dim=1).tolist())
            true_labels.extend(labels.tolist())
    return accuracy_score(true_labels, predictions)

def generate_line_graph(x_data, y_data, x_label, y_label, title, legend_labels, save_path):
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    
    plt.plot(x_data, y_data, linestyle='-', label=legend_labels[0])
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.legend()

    # plt.xticks(np.arange(1,11), x_data, rotation=0)
    
    plt.grid(True)  # Add grid lines
    
    plt.savefig(save_path)

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
    df = pd.read_csv(args.dataset_dir)

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

    for text in texts:
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Store loss and acc for graph plotting
    loss_list, acc_list = [], []

    # Start timing
    start_time = time.time()

    # Training and evaluation loop
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}')

    print(f"Total time consumed (s): {time.time()-start_time}")

    # Plot graph
    model_name = args.model_dir.split('/')[1]
    x_data = [i+1 for i in range(args.epochs)]
    g_title = f"Loss of {model_name} (lr={args.learning_rate})"
    g_path = f"graph_{model_name}_loss.png"
    generate_line_graph(x_data, loss_list, "Epochs", "Loss", g_title, ["loss"], g_path)
    
    g_title = f"Accuracy of {model_name} (lr={args.learning_rate})"
    g_path = f"graph_{model_name}_acc.png"
    generate_line_graph(x_data, acc_list, "Epochs", "Accuracy", g_title, ["accuracy"], g_path)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="models/bert_base_uncased/")
    parser.add_argument('--dataset_dir', type=str, default="datasets/imdb/IMDB.csv")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    
    args = parser.parse_args()
    main(args)
