from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import mlflow

from .model import SimpleCNN
from .config import (
    NUM_EPOCHS,
    LEARNING_RATE,
    RANDOM_STATE,
    THRESHOLD,
    SCHEDULER_PATIENCE,
    EARLY_STOP_PATIENCE,
    BATCH_SIZE,
    NUM_CLASSES,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    PYTORCH_MODEL_PATH
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
def train_cnn_model(train_loader, val_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, device=device, random_seed=RANDOM_STATE):
    """Train the SimpleCNN model on glass vial images."""
    print(f"\n{'='*60}")
    print("TRAINING CNN MODEL")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    model = SimpleCNN().to(device)

    # Weighted BCE-with-logits loss for binary classification. The dataset is
    # imbalanced, so the positive (defective) class is down-weighted by the
    # negative/positive count ratio to rebalance the loss.
    train_labels = train_loader.dataset.labels
    num_positive = int(sum(train_labels))
    num_negative = len(train_labels) - num_positive
    print(f"Training class counts: clean(0)={num_negative}, defective(1)={num_positive}")
    pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32, device=device)
    print(f"Loss pos_weight: {pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Halve the learning rate when validation macro-F1 plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=SCHEDULER_PATIENCE)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_f1': [], 'val_f1': [], 'val_f1_macro': [],
        'best_val_f1_macro': 0.0, 'best_model_state': None,
    }
    epochs_without_improvement = 0

    print("\nStarting training...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            # Forward pass (raw logits)
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics (logits -> probabilities)
            train_running_loss += loss.item()
            probs = torch.sigmoid(logits)
            predicted = (probs >= THRESHOLD).float().view(-1)
            train_total += labels.size(0)
            train_correct += (predicted == labels.view(-1)).sum().item()

            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        epoch_train_loss = train_running_loss / len(train_loader)
        epoch_train_accuracy = train_correct / train_total
        epoch_train_f1 = f1_score(all_train_labels, all_train_preds, average='binary')

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device).view(-1, 1)

                logits = model(images)
                loss = criterion(logits, labels)

                val_running_loss += loss.item()
                probs = torch.sigmoid(logits)
                predicted = (probs >= THRESHOLD).float().view(-1)
                val_total += labels.size(0)
                val_correct += (predicted == labels.view(-1)).sum().item()

                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_accuracy = val_correct / val_total
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='binary')
        epoch_val_f1_macro = f1_score(all_val_labels, all_val_preds, average='macro')

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_accuracy'].append(epoch_train_accuracy)
        history['val_accuracy'].append(epoch_val_accuracy)
        history['train_f1'].append(epoch_train_f1)
        history['val_f1'].append(epoch_val_f1)
        history['val_f1_macro'].append(epoch_val_f1_macro)

        # Step the LR scheduler on validation macro-F1
        scheduler.step(epoch_val_f1_macro)

        # Save best model based on validation macro-F1
        if epoch_val_f1_macro > history['best_val_f1_macro']:
            history['best_val_f1_macro'] = epoch_val_f1_macro
            # Deep-copy the weights so the "best" state does not keep changing
            history['best_model_state'] = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping: no macro-F1 improvement for {EARLY_STOP_PATIENCE} consecutive epochs")
                break

        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}] (lr={optimizer.param_groups[0]['lr']:.6f}):")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_accuracy:.4f} | Train F1: {epoch_train_f1:.4f}")
        print(f"  Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_accuracy:.4f} | Val F1: {epoch_val_f1:.4f} | Val Macro F1: {epoch_val_f1_macro:.4f}")

        # Log to MLflow (only if run is active)
        if mlflow.active_run():
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_train_accuracy, step=epoch)
            mlflow.log_metric("val_accuracy", epoch_val_accuracy, step=epoch)
            mlflow.log_metric("train_f1_score", epoch_train_f1, step=epoch)
            mlflow.log_metric("val_f1_score", epoch_val_f1, step=epoch)
            mlflow.log_metric("val_f1_macro", epoch_val_f1_macro, step=epoch)

    # Load best model weights
    if history['best_model_state'] is not None:
        model.load_state_dict(history['best_model_state'])
        print(f"\nRestored best model weights (best val macro F1: {history['best_val_f1_macro']:.4f})")

    print(f"\nTraining complete!")
    print(f"Best validation macro F1-score: {history['best_val_f1_macro']:.4f}")

    return model, history

from vial_classifier.evaluate import tune_threshold, evaluate_cnn_model
from vial_classifier.data  import (
    load_image_dataset,
    split_dataset,
    create_dataloaders,
    prepare_directories
)

def main():
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 print("Device:", device)
 prepare_directories()

 mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
 mlflow.set_experiment(EXPERIMENT_NAME)

 image_dataset = load_image_dataset() # Load data

 train_dataset, val_dataset, test_dataset = split_dataset(  #Split on train/val/test
        image_dataset,
        random_seed=RANDOM_STATE
    )
 train_loader, val_loader, test_loader = create_dataloaders( #Make dataloaders
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=BATCH_SIZE
    )

 torch.manual_seed(RANDOM_STATE)
 np.random.seed(RANDOM_STATE)

 if mlflow.active_run():                
    mlflow.end_run()
    print("Ended previous active MLflow run")

 with mlflow.start_run(run_name="training_cnn"):

# Log parameters to MLflow
  mlflow.log_param("model_architecture", "SimpleCNN")
  mlflow.log_param("batch_size", BATCH_SIZE)
  mlflow.log_param("num_epochs", NUM_EPOCHS)
  mlflow.log_param("learning_rate", LEARNING_RATE)
  mlflow.log_param("initial_threshold", THRESHOLD)
  mlflow.log_param("num_classes", NUM_CLASSES)
  mlflow.log_param("image_size", "64x64")
  mlflow.log_param("random_seed", RANDOM_STATE)

# Log model architecture details
  model_params = sum(p.numel() for p in SimpleCNN().parameters())
  trainable_params = sum(p.numel() for p in SimpleCNN().parameters() if p.requires_grad)
  mlflow.log_param("total_parameters", model_params)
  mlflow.log_param("trainable_parameters", trainable_params)

  cnn_model, training_history = train_cnn_model(
    train_loader, val_loader,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    device=device,
    random_seed=RANDOM_STATE,
)

# Tune threshold on validation set
  optimal_threshold = tune_threshold(cnn_model, val_loader, device=device)
  mlflow.log_metric("optimal_threshold", optimal_threshold)

# Evaluate on test set with optimal threshold
  test_results = evaluate_cnn_model(cnn_model, test_loader, device=device, threshold=optimal_threshold)
  mlflow.log_metrics({
    "test_accuracy": test_results["accuracy"],
    "test_f1": test_results["f1_score"],
    "test_precision": test_results["precision"],
    "test_recall": test_results["recall"],
})
  torch.save(cnn_model.state_dict(), PYTORCH_MODEL_PATH)
  print(f"Model saved to {PYTORCH_MODEL_PATH}")

 

if __name__ == "__main__":
    main()