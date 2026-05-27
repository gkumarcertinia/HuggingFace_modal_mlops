import wandb
import os
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from huggingface_hub import login
from data import prepare_data
from utils import compute_metrics

def run_training():
    """Initializes and executes the model fine-tuning process pipeline."""
    # Hugging Face authentication
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        raise EnvironmentError("HF_TOKEN environment variable is not set. Please export HF_TOKEN=your_token")

    hf_username = os.environ.get("HF_USERNAME", "your-username")
    hf_repo_id = f"{hf_username}/distilbert-goodreads-genres"

    model_name = 'distilbert-base-cased'
    output_dir = 'distilbert-reviews-genres'
    max_length = 512
    epochs = 3
    batch_size = 16

    print("Extracting and tokenizing datasets...")
    train_dataset, test_dataset, id2label, _ = prepare_data(model_name=model_name, max_length=max_length)

    print("Initializing Sequence Classification Architecture...")
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(id2label)
    )

    # Initialize W&B Run
    wandb.init(
        project="mlops-assignment2",
        name="distilbert-run-1",
        config={
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 3e-5,
            "max_length": max_length,
            "dataset": "UCSD Goodreads",
        }
    )

    training_args = TrainingArguments(
        num_train_epochs=epochs,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        output_dir='./results',
        logging_dir='./logs',
        logging_steps=50,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name="distilbert-run-1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    print("Starting Model Fine-Tuning Optimization Loop...")
    trainer.train()
    
    print(f"Saving finalized checkpoints to: '{output_dir}'")
    trainer.save_model(output_dir)

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    # Push model and tokenizer to Hugging Face Hub
    print(f"Pushing model to Hugging Face Hub: '{hf_repo_id}'...")
    trainer.model.push_to_hub(hf_repo_id)
    tokenizer.push_to_hub(hf_repo_id)
    print(f"Model and tokenizer successfully pushed to: https://huggingface.co/{hf_repo_id}")

    # Log the HF model URL into W&B run summary
    hf_model_url = f"https://huggingface.co/{hf_repo_id}"
    wandb.run.summary["huggingface_model"] = hf_model_url
    print(f"Logged HF model URL to W&B: {hf_model_url}")

    wandb.finish()
if __name__ == '__main__':
    run_training()
