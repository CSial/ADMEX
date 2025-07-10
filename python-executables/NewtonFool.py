import torch
import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import NewtonFool

def denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img = img.copy()
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    return np.clip(img, 0.0, 1.0)

#NewtonFool attack based on ART evasion attack
def run_newtonfool_attack(model, dataloader, device="cpu"):
    model.to(device).eval()

    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=None,
        input_shape=(3, 96, 96),
        nb_classes=2,
        clip_values=(0.0, 1.0)
    )

    all_inputs, all_labels = [], []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        all_inputs.append(inputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    x_test = np.concatenate(all_inputs, axis=0).astype(np.float32)
    y_test = np.concatenate(all_labels, axis=0)
    x_test = np.clip(x_test, 0.0, 1.0)

    attack = NewtonFool(classifier=classifier, max_iter=30, eta=0.01)
    x_adv = attack.generate(x=x_test)

    with torch.no_grad():
        y_pred = torch.argmax(model(torch.tensor(x_test).to(device)), dim=1).cpu().numpy()
        y_adv_pred = torch.argmax(model(torch.tensor(x_adv).to(device)), dim=1).cpu().numpy()

    return x_adv, y_test, y_pred, y_adv_pred
