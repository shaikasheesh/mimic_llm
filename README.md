## Mimic LLM Pre-Training Pipeline

### Overview
This project is a hands-on implementation of the pretraining process used in GPT-style language models.

we leverage:
- Tiktoken for fast GPT-style tokenization

- PyTorch DataLoader with a custom collate_fn for batching and padding

- Block-wise sequence grouping (like real GPT pretraining)

This setup mimics how LLMs are prepared for training, while keeping the code minimal and educational

### Dataset
We used the Cosmopedia 100k dataset from Hugging Face, which contains:

### Pipeline
1. Data Preparation
- Loaded dataset from Hugging Face (datasets.load_dataset)
- Created train/test splits with train_test_split
- Merged prompt + text into a single field

2. Tokenization
- Used tiktoken GPT-2 tokenizer (tiktoken.get_encoding("gpt2"))
- Encoded merged sequences into token IDs

3. Block-wise Grouping
- Concatenated all tokens
- Split into fixed-length blocks (block_size = 512)
- Created inputs (x) and shifted targets (y) for next-token prediction

4. Batching
- Used PyTorch DataLoader for creating batches
- Custom Dataset Function which is inherited from Pytorch Datasets is used to pre-process and prepare the Inputs and targets

5. Training & Evaluation
- Once the dataset is prepared and fed into a DataLoader, we train a GPT-like model using next-token prediction loss (cross-entropy).
  - Inputs (x): a block of block_size tokens
  - Targets (y): the same block, shifted by one token
  - Loss: cross-entropy between model predictions and target tokens
  - Plot: Line plot for train_loss vs test_loss



