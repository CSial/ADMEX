import streamlit as st
import os

def show_session_info():
    st.sidebar.markdown("### Session Overview")
    st.sidebar.markdown("#### Model Info")

    #store info on session state-> needed for attacks, defenses and report generator
    if "uploaded_model" in st.session_state:
        model_status = "Loaded"
        model_path = st.session_state.get("uploaded_model_path", "Unknown")
        model_name = os.path.basename(model_path)
        st.session_state["model_name"] = model_name
        
    else:
        model_status = "❌ Not loaded"
        model_name = "N/A"

    st.sidebar.write(f"Model: {model_status}")
    st.sidebar.write(f"Model File: `{model_name}`")

    if "model_arch" in st.session_state:
        st.sidebar.write(f"Architecture: `{st.session_state['model_arch']}`")

    st.sidebar.markdown("#### Dataset Info")
    if "test_images" in st.session_state:
        img_count = st.session_state["test_images"].shape[0]
        st.sidebar.write(f"Test Samples: `{img_count}`")
        st.session_state["num_samples"] = img_count
        st.session_state["data_split_type"] = "Test"

    else:
        st.sidebar.write("Test Samples: ❌ Not loaded")
        st.session_state["num_samples"] = 0
        st.session_state["data_split_type"] = "Unknown"

    if "test_labels" in st.session_state:
        st.sidebar.write("Labels: Loaded")

    else:
        st.sidebar.write("Labels: ❌ Not loaded")

    
    st.sidebar.markdown("#### Defense Info")

    defense_name = st.session_state.get("preprocessing_method_name", None) 
    last_defense = st.session_state.get("last_defense", None)

    if defense_name:
        st.sidebar.write(f"Preprocessing: `{defense_name.replace('_', ' ').title()}`")

    else:
        st.sidebar.write("Preprocessing: ❌ None")

    if last_defense:
        st.sidebar.write(f"Last Applied Defense: `{last_defense.replace('_', ' ').title()}`")

    if st.sidebar.button("Reset Defense"):
        st.session_state.pop("preprocessing_method", None)
        st.session_state.pop("preprocessing_method_name", None)  
        st.session_state.pop("last_defense", None)
        st.sidebar.success("Defense reset.")

