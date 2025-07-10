import streamlit as st

class Home:
    def __init__(self):
        pass

    def run(self):
        st.markdown("<h1 style='text-align: center;'>Welcome to ADMEX</h1>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style='text-align: center; font-size: 18px;'>
                Welcome to the <strong>Adversarial Model Evaluation eXperimenter (ADMEX)</strong>.<br><br>
                This tool helps you:
                <ul style='text-align: left; margin: 0 auto; max-width: 500px;'>
                    <li>Assess your AI model's robustness.</li>
                    <li>Launch attacks against your model.</li>
                    <li>Apply defenses and compare performance.</li>
                    <li>Visualize performance metrics.</li>
                </ul>
                Use the sidebar on the left to begin.
            </div>
            """,
            unsafe_allow_html=True
        )

       