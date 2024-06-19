import pandas as pd
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 训练函数
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

# 评估函数
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

def main(args):
    # 检查是否安装GPU，如果安装了则使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练的BERT tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(args.model_dir+"tokenizer_config.json")
    model = BertForSequenceClassification.from_pretrained(args.model_dir, num_labels=2).to(device)

    # 加载数据集
    print(f"Loading dataset:{args.dataset_dir}...")
    df = pd.read_csv(args.dataset_dir)
    texts = df['review'][:100].tolist()
    label_mapping = {'positive': 1, 'negative': 0}
    labels = [label_mapping[label] for label in df['sentiment'][:100].tolist()]
    labels = torch.tensor(labels).to(device)

    # 将文本编码为BERT的输入格式
    input_ids = []
    attention_masks = []

    for i, text in enumerate(texts):
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,  # 确保长文本被截断
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    # 将列表转换为张量
    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)

    # 划分训练集和测试集
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, random_state=2024, test_size=0.1)
    train_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2024, test_size=0.1)

    # 创建TensorDataset和DataLoader
    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 定义优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate)

    # 训练和评估循环
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}')

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="./models/bert_base_uncased/")
    parser.add_argument('--dataset_dir', type=str, default="./datasets/imdb/IMDB.csv")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    
    args = parser.parse_args()
    main(args)