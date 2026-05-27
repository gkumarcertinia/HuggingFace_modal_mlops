# DistilBERT Goodreads Genre Classification (MLOps Assignment 2)

This project fine-tunes a `distilbert-base-cased` sequence classification model to predict Goodreads review genres across eight classes (children, comics/graphic, fantasy/paranormal, history/biography, mystery/thriller/crime, poetry, romance, and young adult), then evaluates the model and tracks experiments with Weights & Biases (W&B) while publishing model artifacts (weights + tokenizer) to the Hugging Face Hub for public reuse.

## Setup

1. Clone the repository and move into the project folder.
2. Create and activate a Python virtual environment.
3. Install dependencies from `requirements.txt`.
4. Export your secrets for Hugging Face and W&B.

```bash
git clone https://github.com/gkumarcertinia/HuggingFace_modal_mlops.git
cd HuggingFace_modal_mlops
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export HF_TOKEN=your_huggingface_token
export HF_USERNAME=g25ait2035
export WANDB_API_KEY=your_wandb_api_key
```

## Run Scripts

```bash
python3 data.py      # prepares and caches dataset samples
python3 train.py     # trains model, pushes model/tokenizer to Hugging Face, logs HF URL to W&B summary
python3 eval.py      # evaluates trained checkpoint and logs metrics/artifacts to W&B
```

## Results

| Metric | Score |
|-----------|--------|
| Accuracy | 1.00 |
| F1 Score (weighted) | 1.00 |
| Eval Loss | 0.00185 |

- Hugging Face model: https://huggingface.co/g25ait2035/distilbert-goodreads-genres
- W&B dashboard: https://wandb.ai/g25ait2035-prom-iit-rajasthan/mlops-assignment2

## Repository Contents

- `data.py` — data loading, sampling, tokenization, and dataset building
- `train.py` — fine-tuning + Hugging Face push + W&B summary logging
- `eval.py` — evaluation and report artifact logging
- `utils.py` — dataset wrapper and metric helpers
- `requirements.txt` — Python dependencies
