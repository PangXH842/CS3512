import argparse
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertConfig, BertForSequenceClassification

def load_and_preprocess_data(args):
    # Load dataset
    df = pd.read_csv(args.dataset_dir)
    
    # Split dataset into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Tokenize the datasets
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    
    def tokenize_data(text, labels):
        encodings = tokenizer(
            text.tolist(), 
            padding='max_length', 
            truncation=True, 
            max_length=128, 
            return_tensors='pt'
        )
        label_mapping = {'positive': 1, 'negative': 0}
        labels = torch.tensor([label_mapping[label] for label in labels.tolist()])
        return encodings, labels
    
    train_encodings, train_labels = tokenize_data(train_df['review'], train_df['sentiment'])
    val_encodings, val_labels = tokenize_data(val_df['review'], val_df['sentiment'])
    
    return train_encodings, train_labels, val_encodings, val_labels

def distillation_loss(y_student, y_teacher, labels, alpha=0.5, temperature=2.0):
    """
    y_student: Predictions from the student model
    y_teacher: Predictions from the teacher model
    labels: Ground truth labels
    alpha: Weight for the distillation loss
    temperature: Temperature for the softmax
    """
    # Compute the student loss
    student_loss = F.cross_entropy(y_student, labels)
    
    # Compute the distillation loss
    y_teacher_soft = F.softmax(y_teacher / temperature, dim=-1)
    y_student_soft = F.log_softmax(y_student / temperature, dim=-1)
    distillation_loss = F.kl_div(y_student_soft, y_teacher_soft, reduction='batchmean') * (temperature ** 2)
    
    # Combine the student loss and distillation loss
    total_loss = alpha * student_loss + (1 - alpha) * distillation_loss
    return total_loss

def main(args):
    # Create student model directory
    os.makedirs(args.output_model_dir, exist_ok=True)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.output_model_dir)

    # Load and preprocess the data
    train_encodings, train_labels, val_encodings, val_labels = load_and_preprocess_data(args)

    # Load the teacher model
    teacher_model = BertForSequenceClassification.from_pretrained(args.model_dir)

    # Initialize the student model
    student_config = BertConfig.from_pretrained(args.model_dir, num_labels=2)
    student_model = BertForSequenceClassification(student_config)

    # Create DataLoader
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Optimizer
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.learning_rate)

    # Training loop
    num_epochs = args.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model.to(device)
    student_model.to(device)
    teacher_model.eval()  # Set the teacher model to evaluation mode

    for epoch in range(num_epochs):
        student_model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            # Forward pass for teacher and student
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)

            # Compute the distillation loss
            loss = distillation_loss(student_outputs.logits, teacher_outputs.logits, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item()}')

    print("Knowledge distillation training complete.")

    # Save student model
    student_model.save_pretrained(args.output_model_dir)
    print(f"Student model saved to {args.output_model_dir}")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="models/bert_base_uncased")
    parser.add_argument('--output_model_dir', type=str, default="models/bert_kd")
    parser.add_argument('--dataset_dir', type=str, default="datasets/imdb/IMDB.csv")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    
    args = parser.parse_args()
    main(args)
