import streamlit as st
import numpy as np
import torch
from art.attacks.evasion import SquareAttack
from art.estimators.classification import PyTorchClassifier

def run_square_attack(model, dataloader, device="cpu", eps=0.05):
    model.to(device).eval()

    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 96, 96),
        nb_classes=2,
        clip_values=(0.0, 1.0),
        device_type="cpu"
    )

    all_images, all_labels = [], []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        all_images.append(x_batch.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

    x_test = np.concatenate(all_images, axis=0).astype(np.float32)
    y_test = np.concatenate(all_labels, axis=0)

    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    x_test = (x_test - mean) / std

    
    attack = SquareAttack(
        estimator=classifier,
        eps=eps,
        max_iter=100,
        batch_size=16
    )
    x_adv = attack.generate(x=x_test)

    with torch.no_grad():
        x_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        x_adv_tensor = torch.tensor(x_adv, dtype=torch.float32).to(device)

        y_pred = torch.argmax(model(x_tensor), dim=1).cpu().numpy()
        y_adv_pred = torch.argmax(model(x_adv_tensor), dim=1).cpu().numpy()

    return x_adv, y_test, y_pred, y_adv_pred
