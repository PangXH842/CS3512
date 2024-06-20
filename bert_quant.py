import argparse
import torch
from torch.quantization import quantize_dynamic
from transformers import BertForSequenceClassification, BertTokenizerFast
import os

def main(args):
    # Create output model directory and save tokenizer
    os.makedirs(args.output_model_dir, exist_ok=True)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.output_model_dir)

    # Load the pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()  # Set the model to evaluation mode

    # Perform dynamic quantization
    # The model is prepared for dynamic quantization, and then the JIT is used to compile it.
    quantized_model = quantize_dynamic(
        model,  # Model to be quantized
        {torch.nn.Linear, torch.nn.Embedding}  # Specify the modules to be quantized
    )

    # Save the quantized model
    quantized_model.save_pretrained(args.output_model_dir)
    print(f"Quantized model saved to {args.output_model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="models/bert_base_uncased")
    parser.add_argument('--output_model_dir', type=str, default="models/bert_quant_dynamic")
    
    args = parser.parse_args()
    main(args)