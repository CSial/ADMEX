import streamlit as st
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from art.estimators.classification import PyTorchClassifier
from art.metrics import clever_u
from DeepFool import run_deepfool_attack
from NewtonFool import run_newtonfool_attack
from FGSM import run_fgsm_attack
from PGD import run_pgd_attack
from SquareAttack import run_square_attack

#custom emprirical robustness metric
def compute_custom_empirical_robustness(x_clean, x_adv, y_true, y_adv_pred):
    deltas = x_adv - x_clean
    norms = np.linalg.norm(deltas.reshape(deltas.shape[0], -1), ord=2, axis=1)
    misclassified = y_adv_pred != y_true
    if not np.any(misclassified):
        return np.nan
    avg_norm = norms[misclassified].mean()
    return avg_norm

#custom code to compute loss sensitivity function based on ART sensitivity losss
def custom_loss_sensitivity(model, x, y, device):
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    outputs = model(x_tensor)
    loss = F.cross_entropy(outputs, y_tensor)

    grads = torch.autograd.grad(loss, x_tensor, create_graph=False, retain_graph=False, only_inputs=True)[0]
    grads_flat = grads.view(grads.shape[0], -1)
    norms = grads_flat.norm(p=2, dim=1)
    avg_sensitivity = norms.mean().item()
    return avg_sensitivity

class AttackManager:
    def __init__(self):
        self.model = st.session_state.get("uploaded_model", None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #evaluate the predicted labels for the given tensor
    def batched_predict(self, x_tensor, batch_size=64):
        dataset = TensorDataset(x_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_preds = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, in loader:
                batch_x = batch_x.to(self.device)
                preds = torch.argmax(self.model(batch_x), dim=1).cpu().numpy()
                all_preds.append(preds)
        return np.concatenate(all_preds)
    
    #user selects the attack
    def run(self):
        st.title("Select an Attack")

        if self.model is None:
            st.warning("Please upload and evaluate a model in the baseline step first.")
            return

        st.success("Model found. Ready to run attacks!")
        defense_name = st.session_state.get("preprocessing_method_name", None)
        if defense_name:
            st.info(f"Active Preprocessing Defense: `{defense_name.replace('_', ' ').title()}`")
        else:
            st.info("Active Preprocessing Defense: None")

        attack_type = st.selectbox("Choose an Attack", [
            "DeepFool",
            "Fast Gradient Sign Method",
            "Projected Gradient Descent",
            "NewtonFool",
            "Square Attack"
        ])

        if "test_images" not in st.session_state or "test_labels" not in st.session_state:
            st.warning("No test data found. Please load test data into session state.")
            return

        x_test = st.session_state["test_images"]
        y_test = st.session_state["test_labels"]

        #check for run-time preporecessing defenses
        preproc = st.session_state.get("preprocessing_method_object", None)
        preproc_name = st.session_state.get("preprocessing_method_name", None)
        if preproc and preproc_name:
            try:
                x_test, _ = preproc(x_test)
                st.info(f"{preproc_name.replace('_', ' ').title()} applied to input images.")
            except Exception as e:
                st.warning(f"Failed to apply {preproc_name}: {e}")

        dataloader = DataLoader(TensorDataset(torch.tensor(x_test).float(), torch.tensor(y_test).long()), batch_size=32)
        
        #call the respected attack file
        if st.button("Run Attack"):
            if attack_type == "DeepFool":
                with st.spinner("Running DeepFool Attack..."):
                    adv_imgs, y_true, y_pred, y_adv_pred = run_deepfool_attack(self.model, dataloader, device=self.device)
            elif attack_type == "NewtonFool":
                with st.spinner("Running NewtonFool Attack..."):
                    adv_imgs, y_true, y_pred, y_adv_pred = run_newtonfool_attack(self.model, dataloader, device=self.device)
            elif attack_type == "Fast Gradient Sign Method":
                with st.spinner("Running FGSM Attack..."):
                    adv_imgs, y_true, y_pred, y_adv_pred = run_fgsm_attack(self.model, x_test, y_test, device=self.device)
            elif attack_type == "Projected Gradient Descent":
                with st.spinner("Running PGD Attack..."):
                    adv_imgs, y_true, _, _ = run_pgd_attack(self.model, x_test, y_test, device=self.device)
                    y_pred = self.batched_predict(torch.tensor(x_test).float())
                    y_adv_pred = self.batched_predict(torch.tensor(adv_imgs).float())
            elif attack_type == "Square Attack":
                with st.spinner("Running Square Attack..."):
                    adv_imgs, y_true, y_pred, y_adv_pred = run_square_attack(self.model, dataloader, device=self.device)
            else:
                st.warning("Invalid attack selection.")
                return

            st.session_state["last_attack"] = attack_type
            self.display_results(x_test, adv_imgs, y_true, y_pred, y_adv_pred)
    
    #display the attack result metrics
    def display_results(self, x_test, adv_imgs, y_true, y_pred, y_adv_pred):
        model = self.model.to(self.device).eval()
        adv_imgs_tensor = torch.tensor(adv_imgs).float().to(self.device)
        y_true_tensor = torch.tensor(y_true).long().to(self.device).view(-1)

        with torch.no_grad():
            outputs = model(adv_imgs_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence = probs.max(dim=1)[0]
            
            #cross entropy loss
            adv_loss = F.cross_entropy(outputs, y_true_tensor, reduction='sum').item() / len(y_true_tensor)

        #call tha baseline predictions from session
        original_conf = st.session_state.get("baseline_probs", None)
        conf_drop = (original_conf - confidence.mean().item()) * 100 if original_conf is not None else None

        y_true_np = y_true_tensor.cpu().numpy()
        y_adv_pred = y_adv_pred.cpu().numpy() if isinstance(y_adv_pred, torch.Tensor) else np.array(y_adv_pred)
        correct_adv = np.sum(y_adv_pred == y_true_np)
        #custom accuracy
        adv_accuracy = (correct_adv / len(y_true_np)) * 100.0

        st.session_state["adv_accuracy"] = adv_accuracy
        st.session_state["adv_loss"] = adv_loss
        st.session_state["confidence_drop"] = conf_drop

        st.info(f"Adversarial Accuracy: {adv_accuracy:.2f}%")
        st.info(f"Adversarial Loss: {adv_loss:.4f}")
        if conf_drop is not None:
            st.info(f"Confidence Drop: {conf_drop:.2f}%")
        
        #call custom empirical robustness
        emp_rob = compute_custom_empirical_robustness(x_test, adv_imgs, y_true_np, y_adv_pred)
        st.session_state["empirical_robustness"] = emp_rob
        if not np.isnan(emp_rob):
            st.info(f"Custom Empirical Robustness: {emp_rob:.4f}")
        else:
            st.info(f"Custom Empirical Robustness: All adversarial samples classified correctly.")

        classifier = PyTorchClassifier(
            model=model,
            loss=F.cross_entropy,
            input_shape=(3, 96, 96),
            nb_classes=len(np.unique(y_true_np)),
            clip_values=(0.0, 1.0),
            channels_first=True,
            device_type="gpu" if torch.cuda.is_available() else "cpu"
        )
        
        #CLEVER L2 Score
        try:
            x_first_img = np.asarray(x_test[0], dtype=np.float32)
            clever_score = clever_u(classifier, x_first_img, nb_batches=1, batch_size=1, radius=1.0, norm=2)
            st.session_state["clever_score"] = clever_score
            st.info(f"CLEVER L2 Score: {clever_score:.4f}")
        except Exception as e:
            st.warning(f"CLEVER calculation failed: {e}")
        
        #custom loss sensitivity
        try:
            custom_loss_sens = custom_loss_sensitivity(
                model, x_first_img[np.newaxis, ...], np.array([y_true_np[0]]), self.device
            )
            st.session_state["loss_sensitivity"] = custom_loss_sens
            st.info(f"Custom Loss Sensitivity: {custom_loss_sens:.4f}")
        except Exception as e:
            st.warning(f"Custom Loss Sensitivity calculation failed: {e}")
