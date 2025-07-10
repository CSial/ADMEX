import torch
import numpy as np
import streamlit as st
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

#PGD attack based on ART evasion attack
def run_pgd_attack(model, x_test, y_test, device="cpu", eps=0.1, alpha=0.01, iters=40):
    model.to(device).eval()

    x_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    #y_tensor = torch.tensor(y_test, dtype=torch.long).to(device)-> not needed for untargeted attacks

    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 96, 96),
        nb_classes=2,
        clip_values=(0.0, 1.0)
    )

    attack = ProjectedGradientDescent(
        estimator=classifier,
        norm=np.inf,
        eps=eps,
        eps_step=alpha,
        max_iter=iters,
        targeted=False,
        batch_size=32
    )

    x_adv = attack.generate(x=x_tensor.detach().cpu().numpy())

    with torch.no_grad():
        y_pred = torch.argmax(model(x_tensor), dim=1).cpu().numpy()
        y_adv_pred = torch.argmax(model(torch.tensor(x_adv).to(device)), dim=1).cpu().numpy()

    return x_adv, y_test, y_pred, y_adv_pred
