import streamlit as st
import os
import sys
import uuid

from modules.Home import Home
from modules.BaselineEvaluator import BaselineEvaluator
from modules.AttackManager import AttackManager
from modules.DefenseManager import DefenseManager
from modules.RobustnessReport import RobustnessReport
from modules.TestDataUploader import TestDataUploader 
from modules.SessionInfo import show_session_info
from modules.AttackDashboard import AttackDashboard
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(
    page_title="ADMEX – AI Robustness Checker",
    layout="wide",
    initial_sidebar_state="expanded"
)

class Main:
    def __init__(self):
        self.session_id = st.session_state.get("session_id", None)
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
            st.session_state["session_id"] = self.session_id

        self.menu = [
        "Welcome",
        "Upload & Evaluate Baseline",
        "Upload Data",
        "Run Attack",
        "Add Defense",
        "Generate Report",
        "Report Dashboard"
        ]

        self.pages = {
            "Welcome": Home,
            "Upload & Evaluate Baseline": BaselineEvaluator,
            "Upload Data": TestDataUploader,
            "Run Attack": AttackManager,
            "Add Defense": DefenseManager,
            "Generate Report": RobustnessReport,
            "Report Dashboard": AttackDashboard
        }

    def run(self):
        show_session_info()
        st.sidebar.title("ADMEX Robustness Framework")
        st.sidebar.markdown(f"Session: `{self.session_id[:8]}`")

        selection = st.sidebar.radio("Navigate", self.menu)

        PageClass = self.pages.get(selection)
        if PageClass:
            if PageClass == AttackDashboard:
                page = PageClass(folder_path=r"C:\Users\csial\Desktop\Thesis\Tool\project-name\attack-reports")
            else:
                page = PageClass()
            page.run()


if __name__ == "__main__":
    app = Main()
    app.run()
