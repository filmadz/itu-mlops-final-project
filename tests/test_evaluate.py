from vial_classifier.evaluate import evaluate_cnn_model
from vial_classifier.evaluate import tune_threshold
from vial_classifier.train import train_cnn_model
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import mlflow
import numpy as np
from vial_classifier.config import NUM_EPOCHS, LEARNING_RATE, RANDOM_STATE, THRESHOLD
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_evaluate():
    train_images = torch.rand(8, 3, 64, 64)
    train_labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    
    train_dataset = TensorDataset(train_images, train_labels)
    train_dataset.labels = train_labels.tolist()
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    val_images = torch.rand(4, 3, 64, 64)
    val_labels = torch.tensor([0, 1, 0, 1])
    
    val_dataset = TensorDataset(val_images, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=2)
    
    model, history = train_cnn_model(
        train_loader,
        val_loader,
        num_epochs=1,
        device="cpu"
    )
    test_images = torch.rand(4, 3, 64, 64)
    test_labels = torch.tensor([0, 1, 0, 1])
        
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=2)

    results=evaluate_cnn_model(model,test_loader,device=device, threshold=THRESHOLD)
    best_threshold=tune_threshold(model,val_loader,device=device)

    assert set(results.keys()) == {
    "accuracy",
    "f1_score",
    "precision",
    "recall",
    "roc_auc",
    "confusion_matrix",
    "classification_report"
     }
    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["f1_score"] <= 1
    assert 0 <= results["precision"] <= 1
    assert 0 <= results["recall"] <= 1

    assert len(results["confusion_matrix"]) == 2
    assert len(results["confusion_matrix"][0]) == 2

    assert 0 <= best_threshold <= 1

