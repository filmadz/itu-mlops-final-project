import torch
from vial_classifier.model import SimpleCNN

def test_model_output():
    model= SimpleCNN()
    images= torch.randn(2,3,64,64)
    output= model(images)
    assert output.shape ==(2,1)