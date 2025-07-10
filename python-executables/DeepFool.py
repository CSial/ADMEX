import torch
import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import DeepFool

def run_deepfool_attack(model, dataloader, device="cpu", eps=0.02):
    model.to(device)
    model.eval()

    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 96, 96),
        nb_classes=2,
        clip_values=(0.0, 1.0),
        device_type="cpu"
    )

    attack = DeepFool(classifier=classifier, max_iter=50, epsilon=eps)

    all_images, all_labels = [], []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        all_images.append(inputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    x_test = np.concatenate(all_images, axis=0).astype(np.float32)
    y_test = np.concatenate(all_labels, axis=0)
    x_adv = attack.generate(x=x_test)

    with torch.no_grad():
        y_pred = torch.argmax(model(torch.tensor(x_test).to(device)), dim=1).cpu().numpy()
        y_adv_pred = torch.argmax(model(torch.tensor(x_adv).to(device)), dim=1).cpu().numpy()

    return x_adv, y_test, y_pred, y_adv_pred
