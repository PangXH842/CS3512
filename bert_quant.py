import torch
import argparse
from transformers import BertForSequenceClassification, BertTokenizer
import os

# Function to quantize the BERT model
def main(args):
    # Load the pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()  # Set the model to evaluation mode

    # Fuse modules (if applicable) - This step might be optional depending on the model structure
    # model = torch.quantization.fuse_modules(model, [['bert', 'classifier.dense', 'classifier.out_proj']], inplace=True)

    # Prepare the model for quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Perform static quantization
    torch.quantization.prepare(model, inplace=True)

    # Here we would normally calibrate the model with a representative dataset, but we'll skip this for simplicity
    # For example:
    # with torch.no_grad():
    #     for batch in calibration_data:
    #         model(batch)

    torch.quantization.convert(model, inplace=True)

    # Save the quantized model
    if not os.path.exists(args.output_model_dir):
        os.makedirs(args.output_model_dir)
    
    model.save_pretrained(args.output_model_dir)
    print(f"Quantized model saved to {args.output_model_dir}")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="models/bert_base_uncased")
    parser.add_argument('--output_model_dir', type=str, default="models/bert_quant")
    
    args = parser.parse_args()
    main(args)
