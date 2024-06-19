import argparse
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizerFast
from transformers import BertConfig, BertForSequenceClassification
from transformers import AdamW
from sklearn.metrics import accuracy_score

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
    
    # Combine the two losses
    return alpha * student_loss + (1 - alpha) * distillation_loss

def main(args):
    # Load pre-trained BERT model and tokenizer
    teacher_model = BertForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)

    # Load dataset
    dataset = load_dataset(args.dataset_dir)

    # Tokenize the Dataset

    def tokenize_function(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define a smaller configuration for the student model
    student_config = BertConfig(
        vocab_size=30522,
        hidden_size=384,  # Smaller hidden size
        num_hidden_layers=6,  # Fewer layers
        num_attention_heads=12,
        intermediate_size=1536  # Corresponding intermediate size
    )

    # Initialize the student model
    student_model = BertForSequenceClassification(student_config)

    # Prepare data loaders
    train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(1000))  # Use a subset for quick experimentation
    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=lambda x: x)

    # Optimizer
    optimizer = AdamW(student_model.parameters(), lr=5e-5)

    # Training loop
    num_epochs = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model.to(device)
    student_model.to(device)
    teacher_model.eval()  # Set the teacher model to evaluation mode

    for epoch in range(num_epochs):
        student_model.train()
        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)

            # Forward pass for teacher and student
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)
            student_outputs = student_model(**inputs)

            # Compute the distillation loss
            loss = distillation_loss(student_outputs.logits, teacher_outputs.logits, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    print("Knowledge distillation training complete.")

    # Prepare the validation loader
    val_dataset = tokenized_datasets['validation']
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=lambda x: x)

    # Evaluation loop
    student_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)

            outputs = student_model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="./models/bert_base_uncased")
    parser.add_argument('--output_model_dir', type=str, default="./models/bert_kd")
    parser.add_argument('--dataset_dir', type=str, default="./datasets/imdb")
    
    args = parser.parse_args()
    main(args)
