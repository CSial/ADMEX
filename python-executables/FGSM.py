import torch
import numpy as np
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

#FGSM attack based on ART evasion attack
def run_fgsm_attack(model, x_test, y_test, device="cpu", epsilon=0.03, batch_size=32):

    model.to(device).eval()
    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        input_shape=(3, 96, 96),
        nb_classes=len(np.unique(y_test)),
        clip_values=(0.0, 1.0),
        channels_first=True,
        device_type="cpu"
    )

    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    x_adv = attack.generate(x=x_test)

    y_pred = np.argmax(classifier.predict(x_test, batch_size=batch_size), axis=1)
    y_adv_pred = np.argmax(classifier.predict(x_adv, batch_size=batch_size), axis=1)

    return x_adv, y_test, y_pred, y_adv_pred
