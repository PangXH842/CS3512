import torch
import argparse
from transformers import BertForSequenceClassification, BertTokenizerFast
import os
import torch.nn.utils.prune as prune

# Function to apply pruning to the BERT model
def main(args):
    # Create output model directory and save tokenizer
    os.makedirs(args.output_model_dir, exist_ok=True)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.output_model_dir)

    # Load the pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()  # Set the model to evaluation mode

    # Specify the parameters to prune (e.g., weights in linear layers)
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    # Apply global pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=args.amount,
    )

    # Remove the pruning reparameterization so the model can be saved
    for module, param in parameters_to_prune:
        prune.remove(module, 'weight')

    # Save the pruned model
    model.save_pretrained(args.output_model_dir)
    print(f"Pruned model saved to {args.output_model_dir}")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="models/bert_base_uncased")
    parser.add_argument('--output_model_dir', type=str, default="models/bert_pruned")
    parser.add_argument('--amount', type=float, default=0.2, help="Proportion of connections to prune (0 to 1).")
    
    args = parser.parse_args()
    main(args)
