import streamlit as st

class Information:
    def run(self):
        st.markdown("<h1 style='text-align: center;'>Χρήσιμες Πληροφορίες</h1>", unsafe_allow_html=True)
        st.markdown(
            """"
                <div style='text-align: center; font-size: 18px;'>
                    "Το παρόν αποτελεί μέρος της διπλωματικής εργασίας" 
                    "με τίτλο ¨Μελέτη και Υλοποίηση ενός Συστήματος Αξιολόγησης και Λογοδοσίας Μοντέλων Αυτόματης Λήψης Αποφάσεων¨ " 
                    "της φοιτήτριας Χριστίνα Σιαλμά."
                </div>
            """,
            unsafe_allow_html=True
    )
