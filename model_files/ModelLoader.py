import torch.nn as nn
import torchvision.models as models

#create the model with the selected arch upon method call
class PCamModelFactory:
    def __init__(self, model_name="resnet18", pretrained=True, num_classes=2):
        self.model_name = model_name.lower()
        self.pretrained = pretrained
        self.num_classes = num_classes

    def create_model(self):
        if self.model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if self.pretrained else None
            model = models.resnet18(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif self.model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if self.pretrained else None
            model = models.resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

        return model
