import torch.nn as nn
import torchvision.models as models

#method called from TrainModel.py to create the model
def create_resnet18_for_pcam():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model
