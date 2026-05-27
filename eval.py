import json
import wandb
from transformers import DistilBertForSequenceClassification, Trainer
from sklearn.metrics import classification_report
from data import prepare_data
from utils import compute_metrics

def run_evaluation():
    """Validates the frozen checkpoint against validation sets and logs results."""
    model_dir = 'distilbert-reviews-genres'
    
    wandb.init(project="mlops-assignment2", name="evaluation-run", job_type="eval")

    # Retrieve preprocessed splits
    _, test_dataset, id2label, test_labels = prepare_data()

    print(f"Loading cached model from: '{model_dir}'")
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)

    trainer = Trainer(model=model, compute_metrics=compute_metrics)

    print("Evaluating loss and metrics configurations...")
    eval_metrics = trainer.evaluate(test_dataset)
    print("Test Set Metrics Evaluation Summary:", eval_metrics)

    # Handle direct log keys mapping safely
    wandb.log({
        "final/loss": eval_metrics.get("eval_loss"),
        "final/accuracy": eval_metrics.get("eval_accuracy"),
        "final/f1": eval_metrics.get("eval_f1"),
    })


    print("Extracting predictions...")
    predictions_output = trainer.predict(test_dataset)
    predicted_ids = predictions_output.predictions.argmax(-1).flatten().tolist()
    predicted_labels = [id2label[idx] for idx in predicted_ids]

    # Full per-class classification report
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    metrics_report_dict = classification_report(
        test_labels, predicted_labels,
        target_names=target_names,
        output_dict=True
    )
    metrics_report = classification_report(
        test_labels, predicted_labels,
        target_names=target_names
    )
    weighted_f1 = metrics_report_dict.get("weighted avg", {}).get("f1-score")
    if weighted_f1 is not None:
        wandb.log({"final/weighted_f1": weighted_f1})
    print("\n--- Structural Classification Report ---")
    print(metrics_report)

    # Save artifact outputs locally
    # eval_report.json  — full classification report dict (per assignment reference)
    with open('eval_report.json', 'w') as f:
        json.dump(metrics_report_dict, f, indent=2)

    # classification_report.txt — human-readable text version
    with open('classification_report.txt', 'w') as f:
        f.write(metrics_report)

    # eval_metrics.json — raw Trainer evaluate() output
    with open('eval_metrics.json', 'w') as f:
        json.dump(eval_metrics, f, indent=4)

    print("Evaluation tracking assets written successfully to workspace outputs.")

    # Upload versioned Artifact into W&B (both report files)
    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file("eval_report.json")          # classification report dict
    artifact.add_file("classification_report.txt") # human-readable text
    artifact.add_file("eval_metrics.json")          # trainer metrics
    wandb.log_artifact(artifact)
    
    wandb.finish()

if __name__ == '__main__':
    run_evaluation()
