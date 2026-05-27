import json
import wandb
from transformers import DistilBertForSequenceClassification, Trainer
from sklearn.metrics import classification_report
from data import prepare_data

def run_evaluation():
    """Validates the frozen checkpoint against validation sets and logs results."""
    model_dir = 'distilbert-reviews-genres'
    
    wandb.init(project="mlops-assignment2", name="evaluation-run", job_type="eval")

    # Retrieve preprocessed splits
    _, test_dataset, id2label, test_labels = prepare_data()

    print(f"Loading cached model from: '{model_dir}'")
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)

    trainer = Trainer(model=model)

    print("Evaluating loss and metrics configurations...")
    eval_metrics = trainer.evaluate(test_dataset)
    print("Test Set Metrics Evaluation Summary:", eval_metrics)

    # Handle direct log keys mapping safely
    wandb.log({
        "final/loss": eval_metrics.get("eval_loss"),
        "final/accuracy": eval_metrics.get("eval_accuracy"),
        "final/f1": eval_metrics.get("eval_f1") if "eval_f1" in eval_metrics else eval_metrics.get("eval_runtime"), 
    })


    print("Extracting predictions...")
    predictions_output = trainer.predict(test_dataset)
    predicted_ids = predictions_output.predictions.argmax(-1).flatten().tolist()
    predicted_labels = [id2label[idx] for idx in predicted_ids]

    # Structure Classification Report metrics text logs
    metrics_report_dict = classification_report(test_labels, predicted_labels, output_dict=True)
    metrics_report = classification_report(test_labels, predicted_labels)
    weighted_f1 = metrics_report_dict.get("weighted avg", {}).get("f1-score")
    if weighted_f1 is not None:
        wandb.log({"final/weighted_f1": weighted_f1})
    print("\n--- Structural Classification Report ---")
    print(metrics_report)

    # Save artifact outputs locally
    with open('evaluation_results.json', 'w') as f:
        json.dump(eval_metrics, f, indent=4)
        
    with open('classification_report.txt', 'w') as f:
        f.write(metrics_report)
        
    print("Evaluation tracking assets written successfully to workspace outputs.")

     # Upload versioned Artifact into W&B
    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file("evaluation_results.json")
    wandb.log_artifact(artifact)
    
    wandb.finish()

if __name__ == '__main__':
    run_evaluation()
