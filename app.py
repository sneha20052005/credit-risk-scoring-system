
import streamlit as st
import pandas as pd
from pathlib import Path

# Configure the dashboard page
st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="📊",
    layout="wide"
)

# Display the dashboard title
st.title("Credit Risk Scoring Dashboard")
st.write(
    "An overview of model performance "
    "and predicted customer risk."
)

with st.expander("About this project"):
    st.markdown("""
    ### Objective
    Predict the likelihood of customers experiencing
    payment difficulties using application data.

    ### Models evaluated
    - Logistic Regression: baseline model
    - LightGBM: improved tree-based model

    ### Current evaluation
    - LightGBM ROC-AUC: 0.7608
    - LightGBM KS Score: 39.22

    ### Limitations
    - The ROC-AUC target of 0.80 has not yet been reached.
    - Risk bands are illustrative, not approved lending cutoffs.
    - Group-level risk differences require further fairness analysis.
    """)


# Locate the saved output files
output_dir = Path(__file__).parent / "outputs"
predictions_file = output_dir / "dashboard_predictions.csv"
risk_summary_file = output_dir / "risk_band_summary.csv"

# Check that the required files exist
if predictions_file.exists() and risk_summary_file.exists():

    # Load the saved results
    dashboard_df = pd.read_csv(predictions_file)
    risk_summary = pd.read_csv(risk_summary_file)

    # Show model performance
    col1, col2 = st.columns(2)

    with col1:
        st.metric("LightGBM ROC-AUC", "0.7608")

    with col2:
        st.metric("LightGBM KS Score", "39.22")

    st.info(
        "Baseline evaluation: KS target achieved; "
        "ROC-AUC still needs improvement."
    )

    # Display the customer risk distribution
    st.subheader("Customer Risk Distribution")

    st.bar_chart(
        risk_summary.set_index("Risk Band")["Customer Count"]
    )

    # Display the percentage in each risk band
    st.subheader("Risk Band Summary")

    st.dataframe(risk_summary, width="stretch")

    # Display a sample of customer predictions
    st.subheader("Sample Customer Predictions")

    st.dataframe(dashboard_df.head(10), width="stretch")
    
    # Show the most important model features
    st.subheader("Top 15 Important Features")

    importance_file = output_dir / "feature_importance.csv"

    if importance_file.exists():
        importance_df = pd.read_csv(importance_file)

        st.bar_chart(
            importance_df.set_index("Feature")["Importance"],
            horizontal=True,
            width="stretch"
        )
    else:
        st.warning(
            "Feature importance file not found. "
            "Please save it from the notebook first."
        )
    
    # Compare predicted risk across gender groups
    st.subheader("Fairness Check: Average Predicted Risk by Gender")

    fairness_file = output_dir / "fairness_comparison.csv"

    if fairness_file.exists():
        fairness_df = pd.read_csv(fairness_file)

        st.bar_chart(
            fairness_df.set_index("gender")["Average_Predicted_Risk"],
            width="stretch"
        )

        st.caption(
            "This is a preliminary group comparison, not proof "
            "that the model is fair or unfair."
        )
    else:
        st.warning(
            "Fairness results not found. Save them from the notebook first."
        )


else:
    st.warning(
        "Dashboard data files were not found. "
        "Please save the notebook results in the project's outputs folder."
    )
