import argparse
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizerFast
import os

class FactorizedClassifier(nn.Module):
    def __init__(self, original_layer, factor_size):
        super(FactorizedClassifier, self).__init__()
        self.factor_size = factor_size
        # Store original layer for reference
        self.original_layer = original_layer
        
        # Create two new layers for factorization
        self.linear1 = nn.Linear(original_layer.in_features, factor_size)
        self.linear2 = nn.Linear(factor_size, original_layer.out_features)
        
        # Initialize weights and biases
        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear1.bias.data.zero_()
        nn.init.xavier_uniform_(self.linear2.weight)
        self.linear2.bias.data.copy_(self.original_layer.bias.data)
        
    def forward(self, x):
        # Apply the original layer first to get the initial output
        x = self.original_layer(x)
        # Apply factorization
        x = self.linear1(x)
        x = self.linear2(x)
        return x

def factorize_model(model, factor_size):
    # Iterate through all layers and replace with factorized version if applicable
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and module != model.bert.pooler:
            factorized_layer = FactorizedClassifier(module, factor_size)
            # Replace the original layer with the factorized layer
            setattr(model, name, factorized_layer)
    return model

def main(args):
    # Create output model directory and save tokenizer
    os.makedirs(args.output_model_dir, exist_ok=True)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.output_model_dir)

    # Load the pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()
    
    # Factorize the model
    factorized_model = factorize_model(model, args.factor_size)
    
    # Save the factorized model
    model.save_pretrained(args.output_model_dir)
    print(f"Factorized model saved to {args.output_model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="models/bert_base_uncased")
    parser.add_argument('--output_model_dir', type=str, default="models/bert_wfact")
    parser.add_argument('--factor_size', type=int, default=128, help="Size of the factor in the factorized layers")
    
    args = parser.parse_args()
    main(args)