import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

#define all the reports for the attack dashboard.
class AttackDashboard:
    def __init__(self, folder_path):
        self.folder_path = folder_path
    
    #get defense from model name or from Defense column in csv
    def get_defense(self, row):
        if pd.notna(row.get("Defense")) and row["Defense"] not in ["", None]:
            if str(row["Defense"]).strip().lower() == "none":
                pass
            else:
                return row["Defense"]
            
        name = str(row.get("Model", "")).lower()
        if "fgsm" in name:
            return "FGSM"
        
        elif "pgd" in name:
            return "PGD"
        
        return "None"

    def run(self):
        st.title("Attack Dashboard")

        #current implementation only supports csv files, can be modified to support more
        report_files = [f for f in os.listdir(self.folder_path) if f.endswith(".csv")]
        if not report_files:
            st.warning("No reports found in the specified folder.")
            return

        df_list = [pd.read_csv(os.path.join(self.folder_path, f)) for f in report_files]
        df = pd.concat(df_list, ignore_index=True)

        if "Date" in df.columns:
            df.rename(columns={"Date": "Timestamp"}, inplace=True)

        for col in [
            "Clean Accuracy (%)", 
            "Adversarial Accuracy (%)", 
            "Accuracy Drop (%)",
            "Adversarial Loss", 
            "Empirical Robustness", 
            "CLEVER L2 Score", 
            "Custom Loss Sensitivity"
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Defense"] = df.apply(self.get_defense, axis=1)

        # reformat defense name to be more user friendly pattern
        df["Defense"] = df["Defense"].str.strip().str.lower().replace({
            "fgsm": "FGSM", 
            "pgd": "PGD", 
            "feature_squeezing": "Feature Squeezing",
            "feature squeezing": "Feature Squeezing", 
            "jpeg_compression": "JPEG Compression",
            "jpeg compression": "JPEG Compression", 
            "label_smoothing": "Label Smoothing",
            "label smoothing": "Label Smoothing", 
            "none": "None"
        })

        #allow user to filter with attack
        st.subheader("Filter Options")
        attacks = ["All"] + sorted(df["Attack"].dropna().unique().tolist())
        selected_attack = st.selectbox("Select Attack", attacks, index=0)

        #allow user to filter with defense
        defenses = ["All"] + sorted(df["Defense"].dropna().unique().tolist())
        selected_defense = st.selectbox("Select Defense", defenses, index=0)

        filtered_df = df.copy()
        if selected_attack != "All":
            filtered_df = filtered_df[filtered_df["Attack"] == selected_attack]

        if selected_defense != "All":
            filtered_df = filtered_df[filtered_df["Defense"] == selected_defense]

        #print out all csv contents under specified path
        st.subheader("Attack Summary")
        st.dataframe(filtered_df)
        avg_clean = filtered_df["Clean Accuracy (%)"].mean()
        avg_adv = filtered_df["Adversarial Accuracy (%)"].mean()
        avg_drop = filtered_df["Accuracy Drop (%)"].mean()

        #loss, robustness, CLEVER L2, and custom loss are attack based and filter can selected filters apply to them
        avg_loss = filtered_df["Adversarial Loss"].mean() if "Adversarial Loss" in filtered_df.columns else None
        avg_rob = filtered_df["Empirical Robustness"].mean() if "Empirical Robustness" in filtered_df.columns else None
        avg_clever = filtered_df["CLEVER L2 Score"].mean() if "CLEVER L2 Score" in filtered_df.columns else None
        avg_sens = filtered_df["Custom Loss Sensitivity"].mean() if "Custom Loss Sensitivity" in filtered_df.columns else None
        
        #model metrics summary
        if avg_clean is not None:
            st.markdown(f"**Average Clean Accuracy:** <span style='color:green'>{avg_clean:.2f}%</span>", unsafe_allow_html=True)
        if avg_adv is not None:
            st.markdown(f"**Average Adversarial Accuracy:** <span style='color:green'>{avg_adv:.2f}%</span>", unsafe_allow_html=True)
        if avg_drop is not None:    
            st.markdown(f"**Average Accuracy Drop:** <span style='color:crimson'>{avg_drop:.2f}%</span>", unsafe_allow_html=True)
        if avg_loss is not None:
            st.markdown(f"**Average Adversarial Loss:** <span style='color:crimson'>{avg_loss:.4f}</span>", unsafe_allow_html=True)
        if avg_rob is not None:
            st.markdown(f"**Average Empirical Robustness:** <span style='color:blue'>{avg_rob:.4f}</span>", unsafe_allow_html=True)
        if avg_clever is not None:
            st.markdown(f"**Average CLEVER L2 Score:** <span style='color:blue'>{avg_clever:.4f}</span>", unsafe_allow_html=True)
        if avg_sens is not None:
            st.markdown(f"**Average Custom Loss Sensitivity:** <span style='color:blue'>{avg_sens:.4f}</span>", unsafe_allow_html=True)

        #accuracy drop heatmap
        st.subheader("Accuracy Drop by Attack")
        bar_chart = alt.Chart(filtered_df).mark_bar().encode(
            x=alt.X("Attack", sort="-y"),
            y="Accuracy Drop (%)",
            color=alt.Color("Accuracy Drop (%)", scale=alt.Scale(scheme='redblue')),
            tooltip=["Attack", "Accuracy Drop (%)", "Defense"]
        ).properties(width=700, height=400)
        st.altair_chart(bar_chart, use_container_width=True)

        st.subheader("Accuracy Drop Heatmap (Attack vs Defense)")
        pivot_drop = df.pivot_table(values="Accuracy Drop (%)", index="Attack", columns="Defense", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_drop, annot=True, fmt=".1f", cmap="coolwarm", cbar_kws={'label': 'Drop (%)'})
        plt.gca().invert_yaxis()  
        st.pyplot(fig)

        #emprical robustness heatmap
        if "Empirical Robustness" in df.columns:
            st.subheader("Empirical Robustness Heatmap")
            pivot_rob = df.pivot_table(values="Empirical Robustness", index="Attack", columns="Defense", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_rob, annot=True, fmt=".1f", cmap="Greens", cbar_kws={'label': 'Empirical Robustness'})
            plt.gca().invert_yaxis()  
            st.pyplot(fig)

        #CLEVER L2 heatmap
        if "CLEVER L2 Score" in df.columns:
            st.subheader("CLEVER L2 Heatmap")
            pivot_clever = df.pivot_table(values="CLEVER L2 Score", index="Attack", columns="Defense", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_clever, annot=True, fmt=".2f", cmap="Purples", cbar_kws={'label': 'CLEVER L2'})
            plt.gca().invert_yaxis()  
            st.pyplot(fig)

        #loss heatmap
        if "Custom Loss Sensitivity" in df.columns:
            st.subheader("Custom Loss Sensitivity Heatmap")
            pivot_sens = df.pivot_table(values="Custom Loss Sensitivity", index="Attack", columns="Defense", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_sens, annot=True, fmt=".4f", cmap="Oranges", cbar_kws={'label': 'Loss Sensitivity'})
            plt.gca().invert_yaxis()  
            st.pyplot(fig)


        #defense usage pie chart
        st.subheader("Defense Usage Frequency")
        defense_counts = df["Defense"].fillna("None").value_counts().reset_index()
        defense_counts.columns = ["Defense", "Count"]
        fig, ax = plt.subplots()
        ax.pie(defense_counts["Count"], labels=defense_counts["Defense"], autopct="%1.1f%%", startangle=140)
        ax.axis("equal")
        st.pyplot(fig)
