from model_files import PCamModelFactory
import torch

if __name__ == "__main__":
    factory = PCamModelFactory(model_name="resnet18", pretrained=True)
    model = factory.create_model()
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")
