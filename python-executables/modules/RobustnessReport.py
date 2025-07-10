import streamlit as st
import os
from fpdf import FPDF
import pandas as pd
from datetime import datetime

class RobustnessReport:
    def __init__(self):
        #retrieve all info passed to session state
        self.baseline = st.session_state.get("baseline_metrics", {})
        self.attack = st.session_state.get("last_attack", "N/A")
        self.adv_accuracy = st.session_state.get("adv_accuracy", None)
        self.adv_loss = st.session_state.get("adv_loss", None)
        self.emp_rob = st.session_state.get("empirical_robustness", None)
        self.clever_score = st.session_state.get("clever_score", None)
        self.loss_sensitivity = st.session_state.get("loss_sensitivity", None)
        self.art_clean_acc = self.baseline.get("art_clean_accuracy", None)

        #keep model name for use in report and dashboard
        self.model_name = (
            os.path.basename(st.session_state.get("uploaded_model_name", "Unknown"))
            if "uploaded_model_name" in st.session_state
            else "Unknown"
        )

        self.dataset_type = st.session_state.get("data_split_type", "Unknown")
        self.num_samples = (
            st.session_state["test_images"].shape[0]
            if "test_images" in st.session_state
            else "Unknown"
        )

    #report generator
    def run(self):
        st.header("Robustness Report")

        if not self.baseline:
            st.warning("No baseline data available. Please run a baseline evaluation first.")
            return

        #get metrics from session state
        clean_acc = self.baseline.get("clean_accuracy", 0.0)
        adv_acc = self.adv_accuracy
        adv_loss = self.adv_loss

        #calcuate accuracy drop clean accuracy - adversarial accuracy->this might cause paradox
        drop = clean_acc - adv_acc if adv_acc is not None else None

        #info aboout the model and dataset
        st.subheader("Model & Dataset Information")
        st.markdown(f"**Model**: `{self.model_name}`")
        st.markdown(f"**Data Type**: `{self.dataset_type}`")
        st.markdown(f"**Number of Samples**: `{self.num_samples}`")

        st.subheader("Baseline Performance")
        st.metric("Accuracy (clean)", f"{clean_acc:.2f}%")

        if self.art_clean_acc is not None:
            st.metric("Accuracy (ART Classifier)", f"{self.art_clean_acc:.2f}%")

        if adv_acc is not None:
            st.metric("Accuracy (adversarial)", f"{adv_acc:.2f}%")
        if adv_loss is not None:
            st.metric("Adversarial Loss", f"{adv_loss:.4f}")

        st.subheader("Last Attack")
        st.write(f"{self.attack}")

        st.subheader("Defense Applied")

        #check if defense was applied on run time or if the model name contains one of the two adv defense methods
        runtime_defense = st.session_state.get("preprocessing_method_name", None)
        model_name_lower = self.model_name.lower()
        model_defense = None
        if "fgsm" in model_name_lower:
            model_defense = "FGSM Adversarial Training"
            
        elif "pgd" in model_name_lower:
            model_defense = "PGD Adversarial Training"

        if runtime_defense:
            defense_display = runtime_defense.replace("_", " ").title()

        elif model_defense:
            defense_display = model_defense

        else:
            defense_display = None

        if defense_display:
            st.write(f"{defense_display}")
        else:
            st.write("No defense applied")

        if drop is not None:
            st.markdown(f"Accuracy drop under attack: **{drop:.2f}%**")

        st.subheader("Robustness Metrics")
        if self.emp_rob is not None:
            st.metric("Custom Empirical Robustness (L2 misclassified)", f"{self.emp_rob:.4f}")
        if self.clever_score is not None:
            st.metric("CLEVER L2 Score (first sample)", f"{self.clever_score:.4f}")
        if self.loss_sensitivity is not None:
            st.metric("Custom Loss Sensitivity (first sample)", f"{self.loss_sensitivity:.4f}")

        st.success("Report generation complete.")

        st.markdown("---")
        st.subheader("Export Report")

        export_format = st.radio("Select format", ["PDF", "CSV"], horizontal=True)

        if st.button("Save Report"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            #path hidden for privacy, in future use this can be improved to be dynamic path and not hard coded
            folder = r"C:\Users\csial\Desktop\Thesis\Tool\project-name\attack-reports"
            os.makedirs(folder, exist_ok=True)

            #pass all the columns
            report_data = {
                "Model": self.model_name if self.model_name is not None else "N/A",
                "Dataset Type": self.dataset_type  if self.dataset_type is not None else "N/A",
                "Num Samples": self.num_samples  if self.num_samples is not None else "N/A",
                "Clean Accuracy (%)": clean_acc  if clean_acc is not None else "N/A",
                "ART Clean Accuracy (%)": self.art_clean_acc if self.art_clean_acc is not None else "N/A",
                "Adversarial Accuracy (%)": adv_acc if adv_acc is not None else "N/A",
                "Attack": self.attack  if self.attack is not None else "N/A",
                "Defense": defense_display if defense_display else "None",
                "Accuracy Drop (%)": drop if drop is not None else "N/A",
                "Adversarial Loss": adv_loss if adv_loss is not None else "N/A",
                "Empirical Robustness": self.emp_rob if self.emp_rob is not None else "N/A",
                "CLEVER L2 Score": self.clever_score if self.clever_score is not None else "N/A",
                "Custom Loss Sensitivity": self.loss_sensitivity if self.loss_sensitivity is not None else "N/A",
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            #reports are built with csv file, but pdf is supported as well
            if export_format == "CSV":
                path = os.path.join(folder, f"robustness_report_{timestamp}.csv")
                df = pd.DataFrame([report_data])
                df.to_csv(path, index=False)
                st.success(f"CSV report saved to:\n`{path}`")

            elif export_format == "PDF":
                path = os.path.join(folder, f"robustness_report_{timestamp}.pdf")
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.set_title("Robustness Report")

                pdf.cell(200, 10, txt="Robustness Report", ln=True, align='C')
                pdf.ln(10)
                for k, v in report_data.items():
                    pdf.multi_cell(0, 10, txt=f"{k}: {v}", align='L')

                pdf.output(path)
                st.success(f"PDF report saved to:\n`{path}`")
