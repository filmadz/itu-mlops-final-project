import torch
from torch.utils.data import TensorDataset, DataLoader
from vial_classifier.train import train_cnn_model
from vial_classifier.model import SimpleCNN
from vial_classifier.config import NUM_EPOCHS, LEARNING_RATE, RANDOM_STATE
def test_train_cnn_model():
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

 assert isinstance(model, SimpleCNN)
 assert len(history["train_loss"]) == 1
 assert len(history["val_loss"]) == 1