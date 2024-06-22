import argparse
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertConfig, BertForSequenceClassification
from tqdm import tqdm
import datetime
from torch.cuda.amp import autocast, GradScaler

def log_time(message):
    print(f"{message} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def load_and_preprocess_data(args):
    log_time("Loading dataset")
    df = pd.read_csv(args.dataset_dir)
    
    log_time("Splitting dataset into train and validation sets")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    log_time("Tokenizing the datasets")
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
    
    log_time("Data preprocessing complete")
    return train_encodings, train_labels, val_encodings, val_labels

def distillation_loss(y_student, y_teacher, labels, alpha=0.5, temperature=2.0):
    student_loss = F.cross_entropy(y_student, labels)
    y_teacher_soft = F.softmax(y_teacher / temperature, dim=-1)
    y_student_soft = F.log_softmax(y_student / temperature, dim=-1)
    distillation_loss = F.kl_div(y_student_soft, y_teacher_soft, reduction='batchmean') * (temperature ** 2)
    total_loss = alpha * student_loss + (1 - alpha) * distillation_loss
    return total_loss

def main(args):
    log_time("Creating student model directory")
    os.makedirs(args.output_model_dir, exist_ok=True)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.output_model_dir)

    log_time("Loading and preprocessing the data")
    train_encodings, train_labels, val_encodings, val_labels = load_and_preprocess_data(args)

    log_time("Loading the teacher model")
    teacher_model = BertForSequenceClassification.from_pretrained(args.model_dir)

    log_time("Initializing the student model")
    student_config = BertConfig.from_pretrained(args.model_dir, num_labels=2)
    student_model = BertForSequenceClassification(student_config)

    log_time("Creating DataLoader")
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    log_time("Initializing optimizer")
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)

    scaler = GradScaler()  # For mixed precision training

    log_time("Starting training loop")
    num_epochs = args.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model.to(device)
    student_model.to(device)
    teacher_model.eval()

    accumulation_steps = 2  # Gradient accumulation steps

    for epoch in range(num_epochs):
        log_time(f"Starting epoch {epoch + 1}/{num_epochs}")
        student_model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        
        optimizer.zero_grad()
        for i, batch in enumerate(progress_bar):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            
            with autocast():
                student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
                loss = distillation_loss(student_outputs.logits, teacher_outputs.logits, labels)

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item())

        log_time(f"Epoch {epoch + 1}/{num_epochs} complete with Loss: {loss.item()}")

    log_time("Knowledge distillation training complete.")
    
    log_time("Saving student model")
    student_model.save_pretrained(args.output_model_dir)
    log_time(f"Student model saved to {args.output_model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="bert_base_uncased")
    parser.add_argument('--output_model_dir', type=str, default="models/bert_kd1")
    parser.add_argument('--dataset_dir', type=str, default="IMDB.csv")
    parser.add_argument('--epochs', type=int, default=3)
    
    args = parser.parse_args()
    main(args)
