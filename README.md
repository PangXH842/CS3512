# GKCIII-E BERT Model Compression for Text Classification Project

### Contents:
- evaluate.py
- bert_kd.py
- bert_prune.py
- bert_wfact.py

### Setup:

1. Download the IMDB .csv dataset
2. Download the necessary files for the model bert_base_uncased from Huggingface
3. Arrange the project directories as follows:
        - datasets\imdb
            - IMDB.csv
        - models\bert_base_uncased
            - config.json
            - pytorch_model.bin
            - tokenizer.config.json
            - vocab.txt
        - scripts
            - bert_kd.py
            - bert_prune.py
            - bert_wfact.py
            - evaluate.py

### Procedures

- To train the original BERT model, run `python scripts/evaluate.py` with the default parameters.
- To generate compressed variants of BERT, run one of the following commands: 
    - `python scripts/bert_kd.py` to generate a distilled model;
    - `python scripts/bert_prune.py` to generate a pruned model (unstructured or structured);
    - `python scripts/bert_wfact.py` to generate a weight factorized model.
- The generated models can be used to run `evaluate.py` by altering the `--model_dir` directory.

Script Commands:
- `python scripts/evaluate.py [--model_dir] [--dataset_dir] [--epochs] [--learning rate]`
- `python scripts/bert_kd.py [--model_dir] [--output_model_dir] [--dataset_dir] [--epochs] [--learning rate]` 
- `python scripts/bert_prune.py [--model_dir] [--output_model_dir] [--amount] [--structured]` 
- `python scripts/bert_wfact.py [--model_dir] [--output_model_dir] [--factor_size]` 

