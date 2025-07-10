import streamlit as st
import torch
import os
import numpy as np
from torchvision import models
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader
from art.estimators.classification import PyTorchClassifier
from art.utils import compute_accuracy,to_categorical

#class for uploading and evaluating baseline of model
class BaselineEvaluator:
    def __init__(self):
        self.model = None

    def run(self):
        st.title("Upload your Model & Evaluate its Baseline")
        st.subheader("Upload Your Model")
        uploaded_file = st.file_uploader("Upload a `.pth` model file", type=["pth"])
        model_arch = st.selectbox("Select Model Architecture", [
            "resnet18", "resnet50"
        ])

        if uploaded_file is not None and model_arch:
            save_path = os.path.join("uploaded_models", uploaded_file.name)
            os.makedirs("uploaded_models", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Model saved to: {save_path}")

            try:
                model = self._load_model(save_path, model_arch)
                st.success("Model loaded successfully!")
                self._display_model_summary(model)
                self.model = model

                #save the uploaded model in session state for use in attack and defenses
                st.session_state["uploaded_model"] = model
                st.session_state["model_arch"] = model_arch
                st.session_state["uploaded_model_name"] = uploaded_file.name

                #if images have been uploaded, allow evaluation of baseline
                if "test_images" in st.session_state and "test_labels" in st.session_state:
                    x_test = st.session_state["test_images"]
                    y_test = st.session_state["test_labels"]

                    try:
                        device = torch.device("cpu")
                        model.to(device)
                        model.eval()

                        #check if images are correct format, if not concert to -> channel,height,width
                        if isinstance(x_test, np.ndarray) and x_test.shape[-1] == 3:
                            x_test = np.transpose(x_test, (0, 3, 1, 2))

                        #convert imported data into tensors and create the dataset and data loader 
                        x_tensor = torch.tensor(x_test, dtype=torch.float32)
                        y_tensor = torch.tensor(y_test, dtype=torch.long).squeeze()
                        dataset = TensorDataset(x_tensor, y_tensor)
                        loader = DataLoader(dataset, batch_size=128, shuffle=False)
  
                        all_preds, all_labels = [], []
                        correct, total = 0, 0

                        #for all loaded images get prediction for labels 
                        with torch.no_grad():
                            for images, labels in loader:
                                images, labels = images.to(device), labels.to(device)
                                outputs = model(images)
                                preds = torch.argmax(outputs, dim=1)

                                all_preds.extend(preds.cpu().numpy())
                                all_labels.extend(labels.cpu().numpy())
                                correct += (preds == labels).sum().item()
                                total += labels.size(0)
                        
                        #custom accuracy for correct vs total images
                        accuracy = 100.0 * correct / total

                        #pass outcome to classification report
                        report = classification_report(all_labels, all_preds, target_names=["Normal", "Tumor"])

                        st.success(f"Clean Test Accuracy (PyTorch): **{accuracy:.2f}%**")
                        st.text("Classification Report:")
                        st.code(report, language="text")

                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                        classifier = PyTorchClassifier(
                            model=model,
                            loss=torch.nn.CrossEntropyLoss(),
                            optimizer=optimizer,
                            input_shape=(3, 96, 96),
                            nb_classes=2,
                            clip_values=(0.0, 1.0),
                            channels_first=True,
                            device_type="cpu",
                        )
                        
                        #calculate ART accuracy
                        art_preds = classifier.predict(x_test) 
                        y_test_cat = to_categorical(y_test, nb_classes=2)
                        art_acc, _ = compute_accuracy(art_preds, y_test_cat, abstain=False)
                        st.success(f"Clean Test Accuracy (ART Classifier): **{art_acc * 100:.2f}%**")

                        st.session_state["baseline_metrics"] = {
                            "clean_accuracy": accuracy,
                            "art_accuracy": art_acc,
                            "report": report
                        }

                    except Exception as e:
                        st.error(f"Baseline evaluation failed: {e}")
                else:
                    st.warning("Test data not found in session state. Please upload test images and labels.")

            except Exception as e:
                st.error(f"Failed to load model: {e}")

    #allow users to select the model's arcitecture. For future use cause current code supports resent18!
    def _load_model(self, path, arch):
        if arch == "resnet18":
            model = models.resnet18()
            model.fc = torch.nn.Linear(model.fc.in_features, 2)

        elif arch == "resnet50":
            model = models.resnet50()
            model.fc = torch.nn.Linear(model.fc.in_features, 2)

        else:
            raise ValueError("Unsupported architecture")

        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model

    #if model baseline passed print out model architecture for overview
    def _display_model_summary(self, model):
        if model is None:
            st.error("No model loaded.")
            return

        with st.expander("View Model Architecture"):
            st.code(str(model), language="python")
            num_params = sum(p.numel() for p in model.parameters())
            st.success(f"Total Parameters: {num_params:,}")
