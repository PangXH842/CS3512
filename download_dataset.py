import os
import argparse
from datasets import load_dataset

# Function to download and save dataset
def download_dataset(args):
    dataset_dir = args.dir
    dataset_name = args.name

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # Check if the dataset directory already exists
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if not os.path.exists(dataset_path):
        # Load dataset
        dataset = load_dataset(dataset_name)
        # Save dataset to disk
        dataset.save_to_disk(dataset_path)
        print(f"Dataset {dataset_name} downloaded and saved to {dataset_path}.")
    else:
        print(f"Dataset {dataset_name} already exists at {dataset_path}.")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Download and save a dataset to a local directory.")
    parser.add_argument('--dir', type=str, default="datasets/", help="Directory to save the dataset.")
    parser.add_argument('--name', type=str, default="imdb", help="Name of the dataset.")
    
    args = parser.parse_args()
    download_dataset(args)
