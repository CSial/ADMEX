import streamlit as st
import numpy as np
import torch
import os
from datetime import datetime
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from art.defences.trainer import AdversarialTrainer, AdversarialTrainerMadryPGD
from art.defences.preprocessor import FeatureSqueezing, JpegCompression

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model_files.ModelLoader import PCamModelFactory

#keep path to save the defended model seperately from the trained and uploaded ones
def generate_model_save_path(method_name, base_dir="defended_models"):
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{method_name}_defended_{timestamp}.pth"
    return os.path.join(base_dir, filename)

# only resnet18 implemented, for future use expose to more model architectures
class DefenseManager:
    def __init__(self):
        self.arch = st.session_state.get("model_arch", "resnet18")
        self.device = "cpu"
        self.x_train = st.session_state.get("test_images", None)
        self.y_train = st.session_state.get("test_labels", None)
        self.model = self.load_model()

    def load_model(self):
        if "uploaded_model" not in st.session_state:
            return None
        factory = PCamModelFactory(model_name=self.arch, pretrained=False)
        model = factory.create_model()
        model.load_state_dict(st.session_state["uploaded_model"].state_dict())
        return model

    #allow user to select from 4 defenses
    def run(self):
        st.header("Add Defense Mechanism")

        if self.model is None:
            st.warning("Please upload and evaluate a model in the baseline step first.")
            return

        if self.x_train is None or self.y_train is None:
            st.warning("Training data not loaded. Please load it before applying defenses.")
            return

        defense_type = st.selectbox(
            "Choose Defense Strategy",
            ["Adversarial Training", "Input Preprocessing"],
            key="defense_type"
        )

        if defense_type == "Adversarial Training":
            method = st.selectbox("Training Method", ["FGSM", "PGD"], key="adv_train_method")
        elif defense_type == "Input Preprocessing":
            method = st.selectbox("Preprocessing Method", ["Feature Squeezing", "JPEG Compression"])
        else:
            st.error("Invalid defense type selected.")
            return

        if st.button("Apply Defense"):
            self.apply_defense(defense_type, method)

    #apply selected defense
    def apply_defense(self, defense_type, method):
        path = None

        #if defense is adversarial training model is re-trained and saved localy, user needs to re-load it to evaluate it again
        if defense_type == "Adversarial Training":
            st.info(f"Training model using {method} adversarial training...")
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            classifier = PyTorchClassifier(
                model=self.model,
                loss=torch.nn.CrossEntropyLoss(),
                optimizer=optimizer,
                input_shape=(3, 96, 96),
                nb_classes=2,
                clip_values=(0.0, 1.0),
                channels_first=True,
                device_type="cpu",
            )

            y_train = self.y_train
            if isinstance(y_train, torch.Tensor):
                y_train = y_train.cpu().numpy()
            y_train = np.squeeze(y_train)

            if y_train.ndim == 1 or (y_train.ndim == 2 and y_train.shape[1] == classifier.nb_classes):
                pass
            else:
                raise ValueError(f"Unsupported label shape {y_train.shape}, expected (N,) or (N, {classifier.nb_classes})")
            
            #apply ootb AdversarialTrainer method with attack type = fgsm
            if method == "FGSM":
                attack = FastGradientMethod(estimator=classifier, eps=0.03)
                trainer = AdversarialTrainer(classifier, attacks=attack, ratio=0.5)
                for epoch in range(5):
                    st.info(f"FGSM Adversarial Training Epoch {epoch+1}/5")
                    trainer.fit(self.x_train, y_train)
                path = generate_model_save_path("fgsm")
                torch.save(self.model.state_dict(), path)

            #apply ootb MadryPGD attack method
            elif method == "PGD":
                trainer = AdversarialTrainerMadryPGD(classifier, nb_epochs=5, eps=0.03)
                trainer.fit(self.x_train, y_train)
                path = generate_model_save_path("pgd")
                torch.save(self.model.state_dict(), path)

            else:
                st.error("Unsupported adversarial training method.")
                return

        #preprocessing defenses applicable to image datasets
        elif defense_type == "Input Preprocessing":
            st.info(f"Applying {method} preprocessing defense at inference time...")

            if method == "Feature Squeezing":
                preprocessor = FeatureSqueezing(bit_depth=5, clip_values=(0.0, 1.0))
                defense_name = "feature_squeezing"

            elif method == "JPEG Compression":
                preprocessor = JpegCompression(clip_values=(0.0, 1.0), quality=75)
                defense_name = "jpeg_compression"
            else:
                st.error("Unsupported preprocessing method.")
                return

            #apply preprocessing defense to session state to allow user to select them on attack run
            st.session_state["preprocessing_method"] = preprocessor
            st.session_state["preprocessing_method_name"] = defense_name
            st.session_state["last_defense"] = defense_name
            st.success(f"{method} defense will be applied at inference time.")
            return

        else:
            st.error("Unknown defense type.")
            return

        #save model to path defined above
        if path is not None and os.path.exists(path):
            new_model = self.load_model()
            state_dict = torch.load(path, map_location=self.device)
            new_model.load_state_dict(state_dict)
            st.session_state["uploaded_model"] = new_model
            st.success(f"Model updated and saved at:\n`{path}`")
            st.session_state["last_defense"] = method.lower().replace(" ", "_")
